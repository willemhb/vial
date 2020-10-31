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
from vial.asgi import HTTPResponse, HTTPError
from vial.datastructures import RouteTable

"""
Exception classes.
"""
class VialException(Exception):
    def __init__(self, *args, status=500, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.status = status


class PluginException(VialException):
    pass


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
    def __init__(self, config, routes, plugins, **more_config):
        self.config = config | more_config
        self.routes = RouteTable(routes)
        self.plugins = [ServerErrorPlugin, *plugins, RedirectPlugin, ExceptionPlugin]
        self.app = self._setup_app_stack()

    def _setup_app_stack(self):
        """
        Create a chain of Plugins and applications set up to call each other.
        """
        app = self.routes

        for cls in reversed(self.plugins):
            app = plugin(cls)

        # assert: the final value is a chain of plugins ending at the router
        return app

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
        self.routes = Router(routes)
        self.app = self._stack_plugins()

        while True:
            message = await receive()

            if message['type'] == 'lifespan.startup':
                try:
                    for plugin in self.plugins:
                        await plugin.start()

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
                    for plugin in self.plugins:
                        await plugin.close()

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
    def __init__(self, config, scope, receive, send):
        # Actual ASGI callables
        self.config = config
        self.scope = scope
        self.receive = receive
        self.send = send

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

    def __getitem__(self, key):
        """
        Getitem can be used to access items from the scope.
        """
        return self.scope[key]

    def __setitem__(self, key, value):
        """
        Symmetrical to __getitem__.
        """
        self.scope[key] = value


"""
Application base class.
"""


class Application:
    async def __call__(self, conn):
        raise NotImplementedError("Classes must override this method.")

"""
Endpoint classes.

Endpoints wrap application code and convert their responses into ASGI-compatible events.

Routes are a subclass of endpoints.
"""


class Endpoint(Application):
    """
    Endpoint base class.

    Endpoints wrap regular Python functions (functions that return or 
    yield values) and convert them into appropriate ASGI events.
    """
    async def __call__(self, conn):
        pass


class Route(Endpoint):
    """
    .
    """



class Plugin(Application):
    """
    Plugins are optional components that can be added to an application and either:

    1) Make certain resources or features available when called for.

    2) Modify the scope of an ASGI connection, possibly preempting the rest of
       the application and raising an error.

    Plugins are meant to provide a way of implementing both ASGI middleware
    and a plugin interface similar to Bottles' Plugin API. A Plugin that applies to
    all events of a given scope can be considered middleware.
    """
    def __init_subclass__(cls, **kwargs):
        """
        Set the scopes and events this plugin applies to. The special name '*'
        means 'all scopes' or 'all events'.
        """
        super().__init_subclass__(cls, **kwargs)

        if hasattr(cls, "Config"):
            cls.scopes = getattr(cls.Config, "scopes", ["*"])
            cls.events = getattr(cls.Config, "events", ["*"])
            cls.headers = getattr(cls.Config, "headers", [])

        else:
            cls.scopes = ["*"]
            cls.events = ["*"]
            cls.headers = []

        cls.scopes = set(cls.scopes)
        cls.events = set(cls.events)
        cls.headers = set(map(lambda s: s.encode("utf-8"), cls.headers))


    def __init__(self, app):
        self.app = app

    def check(self, conn):
        """
        Check if this Plugin should be applied to the connection.
        
        This method can be overridden to implement custom logic.
        
        The default check method will return True if the connection
        scope is among the scopes in self and all of its required
        headers (possibly None) are among the connection's headers.
        """
        for s in self.scopes:
            if s == "*" or conn["type"] == s:
                break
        else:
            return False

        for h in self.headers:
            if h not in conn.header_names:
                return False

        else:
            return True


    async def apply(self, conn):
        """
        This method has no default behavior and must be overridden by subclasses.

        All changes to the connection scope should be applied through this method.
        """
        pass

    async def __call__(self, conn):
        """
        This method checks if the plugin should be applied to conn. If yes, it calls
        its apply method. If no, it does nothing. Either way, it calls its inner
        application, passing conn (which may or may not have had its scope, send, or
        receive methods modified by self).

        The __call__ method should be overridden if:

            1) The Plugin wants to raise an Exception when validation fails.

            2) The Plugin wants to wrap the inner application with exception handling.

            3) It wants to do something uncoditionally.

            4) Any combination of the above.
        """
        if self.check(conn):
            await self.apply(conn)

        await self.app(conn)

    async def start(self):
        """
        This method should check that the application the plugin was initialized with
        implements the required scopes and events. It should then initialize the resources
        it needs (for a database plugin, this would be a database connection). 

        If any of the above steps fails, the start method should raise an exception.
        """
        return

    async def close(self):
        """
        This method should clean up any resources it depends on and return None if
        everything went okay. Otherwise it should raise an exception.
        """
        return

"""
Builtin Plugin classes (these are absolutely necessary for an application to run
reliably, and they are included by default).
"""
class ServerErrorPlugin(Plugin):
    pass


class TrustedHostPlugin(Plugin):
    pass


class RedirectPlugin(Plugin):
    """
    Wraps the application-side code with redirect handling.
    """
    class Config:
        scopes = ["http"]

    async def __call__(self, conn):
        try:
            await self.app(conn)

        except HTTPResponse as r:
            if 300 <= r.status <= 309:
                pass

            else:
                raise r



class ExceptionPlugin(Plugin):
    """
    Wraps the application-side code with error handling.
    """
    pass


"""
Authentication and session plugins are also provided as part of the base library.
"""
class SessionPlugin(Plugin):
    pass

class AuthenticationPlugin(Plugin):
    pass

"""
Route decorating helpers.
"""

def route(path, methods):
    """
    Return a wrapper that creates a new Route object for the wrapped function. The wrapper
    returns the undecorated function to enable decorator stacking.
    """
    def wrapper(func):
        pass

    return wrapper


"""
Redirecting functions.

These functions implement simple (300-range) or error-based (400-range)
redirects. They are analogues to Bottles' `redirect` and `abort` functions.
"""
def redirect(url, status=301, headers={}, **more_headers):
    headers = headers | more_headers | {"Location": url}
    raise HTTPResponse(status, headers=headers)


def abort(status, message=None, headers={}, **more_headers):
    headers = headers | more_headers
    

"""
Rendering (templates and static files).
"""


def template(tpl, **ctx):
    """
    Render the template named 'tpl' with the given context variables.
    """


    

"""
Content types.

These Content type classes employ some methods, but mostly exist to support
type hinting the return type from route decorators.
"""


class Text:
    pass


class Json:
    pass


class Html:
    pass


class Xml:
    pass
