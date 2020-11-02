"""
This module isn't really 'extra' so much as 'awaiting implementation'.

Moved here to make working in the main file simpler.
"""


class PluginException(VialException):
    pass



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

# TODO: this

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
