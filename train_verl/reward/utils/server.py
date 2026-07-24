"""Round-robin server manager for the judge model backend.

Reads ``<model_name> -> [ip:port, ...]`` from ``judge_server_routes.json`` and
hands out server addresses in a shuffled round-robin so a single vLLM instance
does not get hammered when many rollouts request judgments concurrently.
"""

import random
import threading

from .request_utils import get_model_servers


class ServerManager:
    """Singleton round-robin dispatcher over the judge servers.

    After every two full passes over the shuffled server list the list is
    reshuffled so long training runs still spread load evenly.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model_servers = get_model_servers()
                    cls._instance._server_state = {}
        return cls._instance

    def _init_or_reshuffle_servers(self, model_name):
        servers = self._model_servers.get(model_name, [])
        if not servers:
            return []
        shuffled = list(servers)
        random.shuffle(shuffled)
        self._server_state[model_name] = {"index": 0, "shuffled_servers": shuffled}
        return shuffled

    def get_next_server(self, model_name):
        """Return the next server for ``model_name`` (or ``None`` if none configured)."""
        if not self._model_servers.get(model_name):
            return None

        with self._lock:
            if model_name not in self._server_state:
                self._init_or_reshuffle_servers(model_name)

            state = self._server_state[model_name]
            shuffled = state["shuffled_servers"]

            # Reshuffle after two full passes.
            if state["index"] >= len(shuffled) * 2:
                self._init_or_reshuffle_servers(model_name)
                state = self._server_state[model_name]
                shuffled = state["shuffled_servers"]

            server = shuffled[state["index"] % len(shuffled)]
            state["index"] += 1
            return server

    def get_server_list(self, model_name):
        return self._model_servers.get(model_name, [])
