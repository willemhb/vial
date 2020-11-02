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
from functools import cached_property
from vial.asgi import ConnectionState, HTTPResponse, HTTPError
from vial.util import RuleTable, MultiDict, NoMatch
from typing import Callable

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
    def __init__(self, config, routes, **more_config):
        self.config = config | more_config
        self.routes = Router(routes)
        self.app = self.routes

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
    def __init__(self, config: dict, scope: dict, receive: Callable, send: Callable):
        # Actual ASGI callables
        self.config = config
        self.scope = scope
        self.receive = receive
        self.send = send

        # Connection state
        self.state = ConnectionState(self.scope)

    @cached_property
    def req_headers(self):
        return self.state.req_headers

    @cached_property
    def resp_headers(self):
        return self.state.resp_headers

    async def __call__(self, event=None, /,):
        """
        Dispatch to either self.receive or self.send, depending on
        the presence or absence of the single (positional-only) event
        argument.
        """
        if event is None:
            return await self.receive()

        else:
            await self.send(event)
            return


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
    def __init__(self, endpoint, route):
        self.endpoint = endpoint
        self._raw_route = route

    def _match_captured_values(self, captured):
        """
        Match the captured path parameters to.
        """
        return dict(zip(filter(None, self.capture_groups), captured))

    async def __call__(self, conn, *captured):
        mapping = self._match_captured_values(captured)
        result = await self.endpoint(conn, *mapping)
        return result


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
        result = await route(conn, *captured)
        await self.handle_result(conn, result)
        return

    async def handle_result(self, conn, result):
        """
        Converts the result from the endpoint into a message and sends
        it over the connection.
        """
        pass

    async def start(self):
        for route in self.routes:
            self._table[route._raw_route] = route

        return

    async def close(self):
        return
