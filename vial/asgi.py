"""
The classes in this module are intended primarily to provide an interface
to the ASGI standard, or otherwise to assist with implementing the spec.
"""
from dataclasses import dataclass
from itertools import starmap
from typing import Union, Iterable, Optional, TypedDict
from enum import Enum
from http.cookies import SimpleCookie
from vial.util import MultiDict

"""
Typedefs.
"""
Json = Union[Iterable, dict, str, bytes, bool, int, float] # ie, serializable
ASGIStr = Union[str, bytes]
RawHeaders = Iterable[Iterable[bytes, bytes]]
Headers = Union[RawHeaders, MultiDict[ASGIStr, Json]]
"""
Exception classes.
"""


class ASGIError(Exception):
    pass


"""
Helper classes.
"""


@dataclass
class ConnectionState:
    """
    Represents all of the state information associated with
    an HTTP connection.
    """
    scope: dict
    # application info
    started: bool = False
    finished: bool = False

    # these attributes hold request data
    req_headers: Headers = MultiDict()
    req_body: bytes = b""
    more_req_body: bool = True

    # these attributes hold response data
    resp_headers: Headers = MultiDict()
    cookies: SimpleCookie = SimpleCookie()
    status: int = 200
    resp_body: bytes = b""
    more_resp_body: bool = True
    received_body_length: int = 0

    def __post_init__(self):
        self.read_request_headers()

    def read_request_headers(self):
        """
        Send the request headers from the scope to self.req_headers.
        """
        decoded = starmap(lambda a, b: a.decode(), b.decode(), self.scope["headers"])
        self.req_headers.update(decoded)

    def read_event(self, event):
        """
        Interpret the event and update the connection state.

        Assumes that invalid events have already been taken care
        of.
        """
        if event["type"] == "http.request":
            if not self.started:
                self.started = True

            self.req_body += event["body"]

            if not event["more_body"]:
                self.more_req_body = False

        elif event["type"] == "http.response.start":
            self.status = event["status"]
            self.headers.update(event["headers"])
            

        elif event["type"] == "http.response.body":
            self.resp_body += event["body"]
            self.received_body_length += len(event["body"])

            if not event["more_body"]:
                self.more_resp_body = False
                self.headers["content-length"] = self.received_body_length
                self.finished = True

        else:
            return


class AsgiTypeKeys(str, Enum):
    """
    Subclasses of this Enum associate ASGI events of a particular type with a specific
    Typed dict callable.
    """
    def __new__(cls, value, evt_dct_cls):
        """
        The actual value is the first argument, the .
        """
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._evt_cls = evt_dct_cls
        return obj



class HTTPResponse(BaseException):
    """
    Making this an Exception class allows it to be 
    raised (useful for redirects) or returned
    like a normal class.
    """
    def __init__(self, status, body="", headers={}, **more_headers):
        self.status = status
        self.body = body
        self.headers = headers | more_headers


class HTTPError(Exception):
    def __new__(cls, status, *args, **kwargs):
        """
        Throw an ASGIError if the status isn't in the error range. 
        """
        if not (400 <= status <= 599):
            raise AsgiError(f"Status {status} out of range for HTTP errors.")

        return super().__new__(cls, status, *args, **kwargs)

    def __init__(self, status, message, headers={}, **more_headers):
        self.status = status
        self.body = message
        self.headers = headers | more_headers


"""
Event classes.
"""


class BaseEvent(dict):
    """
    Use Types Enum to validate keys to validate keys passed to constructor.
    """
    def __new__(cls, **kwargs):
        """
        Try to match the supplied event type to one of the patterns in HTTPTypes.
        """

        if (evt_type := kwargs.get("type", None)) is None:
            raise TypeError("Missing required key 'type'.")

        try:
            return getattr(cls, cls.Types(evt_type)._evt_cls)(**kwargs)

        except ValueError as e:
            raise e


class HTTPEvent(BaseEvent):
    class Types(AsgiTypeKeys):
        REQUEST = "http.request", "BodyEvent"
        RESPONSE_START = "http.response.start", "ResponseEvent"
        RESPONSE_BODY = "http.response.body", "BodyEvent"
        DISCONNECT = "http.disconnect", "Core"

    class Core(TypedDict):
        type: str

    class BodyEvent(Core):
        body: bytes
        more_body: bool

    class ResponseEvent(Core):
        status: int
        headers: Headers


class WebSocketEvent(BaseEvent):
    class Types(AsgiTypeKeys):
        CONNECT = "websocket.connect", "Core"
        ACCEPT = "websocket.accept", "Accept"
        RECEIVE = "websocket.receive", "BodyEvent"
        SEND = "websocket.send", "BodyEvent"
        DISCONNECT = "websocket.disconnect", "CodeEvent"
        CLOSE = "websocket.close", "CodeEvent"

    class Core(TypedDict):
        type: str

    class Accept(Core):
        subprotocol: str
        headers: Headers

    class BodyEvent(Core):
        bytes: bytes
        text: str

    class CodeEvent(Core):
        code: int


class LifespanEvent(BaseEvent):
    class Types(AsgiTypeKeys):
        STARTUP = "lifespan.startup", "Core"
        STARTUP_COMPLETE = "lifespan.startup.complete", "Core"
        STARTUP_FAILED = "lifespan.startup.failed", "Failure"
        SHUTDOWN = "lifespan.shutdown", "Core"
        SHUTDOWN_COMPLETE = "lifespan.shutdown.complete", "Core"
        SHUTDOWN_FAILED = "lifespan.shutdown.failed", "Failure"

    class Core(TypedDict):
        type: str

    class Failure(Core):
        message: str


"""
Scopes classes.
"""


class BaseScope(TypedDict):
    type: str
    asgi: dict[str, str]
    


class HTTPScope(BaseScope):
    http_version: str
    method: str
    scheme: str
    path: str
    raw_path: bytes
    query_string: bytes
    root_path: str
    headers: Headers
    client: tuple[str, int]
    server: tuple[str, int]


class WebsocketScope(BaseScope):
    http_version: str
    method: str
    scheme: str
    path: str
    raw_path: bytes
    query_string: bytes
    root_path: str
    headers: Headers
    client: tuple[str, int]
    server: tuple[str, int]
    subprotocols: Iterable[str]


class LifespanScope(BaseScope):
    pass


"""
Event Handlers are callables that handle the scope-specific logic of a connection.
A connection dispatches a handler classed based on the scope type, raising an exception
if it has no compatible handlers.
"""


class BaseEventHandler:
    """
    Common ancestor of all event handlers.
    """
    def __init__(self, receive, send):
        self._receive = receive
        self._send = send

    async def receive(self):
        raise NotImplementedError("Subclasses most override this method.")

    async def send(self, event):
        raise NotImplementedError("Subclasses must override this method.")


class HTTPEventHandler(BaseEventHandler):
    """
    Handles HTTP messages.

    For now, this class simply returns the entire body as a single chunk for
    both send and receive events.
    """
    def __init__(self, receive, send):
        super().__init__(self, receive, send)

        # Connection specific state
        self._response_start = None
        self._response_body = bytearray()
        self._complete = False
    
    async def receive(self):
        """
        Call self._receive
        """
        event = await self._receive()

        if event["type"] == "http.request":
            body = bytearray()

            while event["more_body"]:
                body.extend(event["body"])

                event = await self._receive()

            else: # Call at least once
                body.extend(event["body"])

            return {
                "type": "http.request",
                "body": bytes(body),
                "more_body": False,
            }

        elif event["type"] == "http.disconnect":
            return event

        else:
            raise ASGIError(f"Unknown http event type {event['type']}")

    async def send(self, event):
        if event["type"] == "http.response.start":
            # Save the response start, but don't send it until the body is received
            self._response_start = event

        elif event["type"] == "http.response.body":
            self._response_body.extend(event["body"])

            if not event["more_body"]:
                # send the start and the body
                await self._send(self._response_start)
                await self._send({
                    "type": "http.response.body",
                    "body": bytes(self._response_body),
                    "more_body": False,
                })
                self._complete = True

            else:
                raise ASGIError(f"Unknown http event type {event['type']}")
