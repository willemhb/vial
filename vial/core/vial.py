"""
Main distribution file for the Vial framework.

Vial is an ASGI-compliant successor to Bottle that prioritizes, 
extensibility, simplicity, and minimal dependencies.

It provides:

    - An extensible root application class, Vial, that can
      run on any server designed to run ASGI applications.

    - A core 'Connection' class, representing the state
      of a client connection (serves the role of request and
      response, but more general).

    - A route decorator, for creating structured routes.

    - A middleware decorator, that registers an action for
      middleware to take for given scopes.

    - A plugin decorator, based on the middleware decorator,
      that conditionally provides resources to the rest of
      the application.

    - An interface to Jinja2 templates.
"""

import sys, os, getopt, inspect
from functools import cached_property
from vial.asgi import (HTTPConnectionState, WebSocketConnectionState,
                       HTTPResponse, HTTPError, HTTPEventHandler,
                       WebSocketEventHandler)
from vial.util import RuleTable, MultiDict, Config, NoMatch, process_route
from typing import Callable, Union, Optional
from urllib.parse import parse_qsl, unquote_plus

"""
Exception classes.
"""
class VialException(Exception):
    def __init__(self, *args, status=500, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.status = status

"""
Main application class.
"""

class Vial:
    """
    Core ASGI application. Its main work is to handle lifespan events and wrap
    all other calls in a Connection object, which is then piped through the
    remaining applications.

    The Vial class also handles lifespan events (most Protocols are handled
    by specialized Event handling subclasses).
    """
    def __init__(self, config, routes):
        self.config = config
        self.router = Router(routes)
        self.app = self.router

    async def __call__(self, scope, receive, send):
        """
        Handle lifespan events, create a new connection, or raise an error.
        """
        if scope['type'] == 'lifespan':
            await self.lifespan(scope, receive, send)

        elif scope['type'] in self.scope_types:
            conn = Connection(self, scope, receive, send)
            await self.app(conn)

        else:
            raise NotImplementedError("Unknown connection scope.")

    async def lifespan(self, scope, receive, send):
        """
        Handle lifespan events (app initialization).
        """
        while True:
            message = await receive()

            if message['type'] == 'lifespan.startup':
                try:
                    await self.router.start()

                except Exception as e:
                    message = {
                        "type": "lifespan.startup.failed",
                        "message": str(e),
                    }

                else:
                    message = {
                        'type': 'lifespan.startup.complete',
                    }

                finally:
                    await send(message)

            elif message['type'] == 'lifespan.shutdown':
                try:
                    await self.router.close()

                except Exception as e:
                    message = {
                        'type': 'lifespan.shutdown.failed',
                        'message': str(e),
                    }

                else:
                    message = {
                        'type': 'lifespan.shutdown.complete',
                    }

                finally:
                    await send(message)
                    return


"""
The connection class.
"""


class Connection:
    """
    The Connection class stores all the components of an ASGI app,
    simplifying the interface. Examples below.

    Standard ASGI:

        async def app(scope, receive, send):
            content_type = scope["headers"]["content-type"]
            event = await receive()
            await send(event)
            ...            

    Using Connection:
        async def app(conn):
            content_type = conn["headers"]["content-type"]
            event = await conn()
            await conn(event)

    async def app(conn):
    """
    HANDLERS = {
        "http": HTTPEventHandler,
        "websocket": WebSocketEventHandler,
    }

    def __init__(self, config: dict, scope: dict, receive: Callable, send: Callable):
        # Actual ASGI callables
        self.config = config
        self.scope = scope
        self.handler = self.HANDLERS[scope["type"]](scope, receive, send)

    @cached_property
    def state(self):
        return self.handler.state

    @cached_property
    def req_headers(self):
        header_pairs = self.scope["headers"].items()
        return MultiDict(starmap(lambda a, b: (a.decode(), b.decode()), header_pairs))

    @cached_property
    def query(self):
        query_string = self.scope["query_string"].decode()
        return MultiDict(parse_qsl(unquote_plus(query_string)))

    async def __call__(self, *args, **kwargs):
        """
        Dispatch to either self.receive or self.send, depending on
        the presence or absence of the single (positional-only) event
        argument.
        """
        if args or kwargs:
            await self.handler.send(*args, **kwargs)

        else:
            return await self.handler.receive()

"""
Application base class.
"""

class Application:
    async def __call__(self, conn):
        raise NotImplementedError("Classes must override this method.")


class Route(Application):
    """
    Route decorator class.

    Extremely bare bones implementation right now, trying to get a basic
    implementation finished.
    """
    def __init__(self, endpoint, route, methods):
        self.endpoint = endpoint
        self._raw_route = route
        self.methods = methods

    async def __call__(self, conn, **captured):
        result = await self.endpoint(conn, **captured)
        return result

    async def start(self):
        """
        Expand the path and set up capturing groups.
        """
        self.route = process_route(self._raw_route)
        return self.route

    async def close(self):
        pass


class Router(Application):
    """
    Instances of the router class manage a RouteTable (implemented using RuleTable)
    whose endpoints are Routes.
    """
    def __init__(self, routes):
        """
        Initialize the RuleTable and set up routes.
        """
        self.routes = routes
        self._table = RuleTable()

    async def __call__(self, conn):
        """
        Try to match the supplied URL to a rule in the router's rule table.
        """
        url = conn.path
        captured, route = self._table[url]
        result = await route(conn, **captured)
        await conn(response)

    async def start(self):
        for route in self.routes:
            expanded_route = await route.start()
            for method in route.methods:
                self._table[(method, expanded_route)] = route

        return self

    async def close(self):
        return


"""
The basic route decorator.
"""




def options(rule: str):
    """
    Shorthand for registering an endpoint that responds to get requests.
    """
    def wrapper(endpoint):
        new_route = Route(endpoint, rule, methods=["OPTIONS"])
        routes = endpoint.__globals__.setdefault("__vial_routes__", [])
        routes.append(new_route)

        return endpoint

    return wrapper


def patch(rule: str):
    """
    Shorthand for registering an endpoint that responds to get requests.
    """
    def wrapper(endpoint):
        new_route = Route(endpoint, rule, methods=["PATCH"])
        routes = endpoint.__globals__.setdefault("__vial_routes__", [])
        routes.append(new_route)

        return endpoint

    return wrapper


def head(rule: str):
    """
    Shorthand for registering an endpoint that responds to get requests.
    """
    def wrapper(endpoint):
        new_route = Route(endpoint, rule, methods=["HEAD"])
        routes = endpoint.__globals__.setdefault("__vial_routes__", [])
        routes.append(new_route)

        return endpoint

    return wrapper


def _gather_routes():
    """
    Search imported modules for registered routes (modules that have a
    __vial_routes__ attribute).

    Return a Router instance from the collected routes. 
    """
    routes = _get_module_scope().get("__vial_routes__", [])
    
    for module in sys.modules.values():
        if "__vial_routes__" in vars(module):
            routes.extend(vars(module)["__vial_routes__"])

    return routes

def _get_module_scope():
    """
    Get the globals for the module where this function is called.
    """
    module_scope = inspect.currentframe().f_back.f_globals
    return {k:v for k,v in module_scope.items() if k != '__builtins__'}


"""
Cli and startup functions.
"""


def parse_cli(so="hvdc:", lo=["help", "verbose", "debug", "confs="]):
    """
    Basic CLI parser.
    """
    opts, args = getopt.gnu_getopt(sys.argv[1:], so, lo)

    d = dict(opts)

    for k in d:
        if d[k] == '':
            d[k] = True

    d["args"] = args

    return d


def run(host: str="localhost", port: int=8000, config: Optional[Config] = None, *, routes: list, **more_config):
    """
    Parse command line arguments, and start the server.
    """
    import uvicorn

    optdict = parse_cli()

    if (a := optdict["args"]): # cli always overrides arguments from the script
        host = a[0]
        port = int(a[1])

    if "help" in optdict:
        print(globals()["__doc__"])
        return

    # TODO: other cli options

    if config is None:
        config = Config(optdict["config"], **more_config)

    app = Vial(config, routes)

    uvicorn.run(app, host=host, port=port)
