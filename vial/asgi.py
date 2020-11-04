"""
The classes in this module are intended primarily to provide an interface
to the ASGI standard, or otherwise to assist with implementing the spec.
"""
from dataclasses import dataclass, field
from itertools import starmap
from typing import Union, Iterable, Optional, TypedDict
from enum import Enum
from http.cookies import SimpleCookie
from vial.util import MultiDict

"""
Typedefs.
"""
Text = Union[str, bytes]
Json = Union[Iterable, dict, str, bytes, bool, int, float] # ie, serializable
ASGIStr = Union[str, bytes]
RawHeaders = Iterable[Iterable[bytes]]
Headers = Union[RawHeaders, MultiDict]
"""
Exception classes.
"""


class ASGIError(Exception):
    pass


"""
Helper classes.
"""


@dataclass
class HTTPConnectionState:
    """
    Represents all of the state information associated with
    an HTTP connection.
    """
    # where are we in the request/response cycle?
    started: bool = False
    resp_started: bool = False
    finished: bool = False

    # These attributes hold information about the request body
    # (this information is not contained in the scope)
    req_body: bytes = b""
    more_req_body: bool = True

    # these attributes hold response data
    headers: Headers = field(default_factory=MultiDict)
    cookies: SimpleCookie = field(default_factory=SimpleCookie)
    status: int = 200
    body: bytes = b""
    more_body: bool = True

    def encode_headers(self):
        """
        Re-encode the response headers to be sent to the server in a send event.
        """
        return [(k.encode('utf8'), v.encode('utf8')) for k, v in self.headers.items()]


@dataclass
class WebSocketConnectionState:
    """
    Represents all of the state information associated with
    a WebSocket connection.
    """
    # 
    accepted: bool = False
    closed: bool = False

    # These attributes hold information about the request body
    # (this information is not contained in the scope)
    subprotocol: Optional[str] = None
    # these attributes hold response data
    headers: Headers = MultiDict()
    code: int = 1000

    def encode_headers(self):
        """
        Re-encode the response headers to be sent to the server in a send event.
        """
        return [(k.encode('utf8'), v.encode('utf8')) for k, v in self.headers.items()]


class AsgiTypeKeys(str, Enum):
    """
    Subclasses of this Enum associate ASGI events of a particular type with a specific
    Typed dict callable.
    """
    @classmethod
    def typekeys(cls):
        return set(s.value for s in cls)


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
Event & scope classes.
"""


class BaseEvent(TypedDict):
    """
    Minimal set of keys required in an event.
    """
    type: str


class HTTPEvent(BaseEvent, total=False):
    body: bytes
    more_body: bool
    status: int
    headers: RawHeaders


class HTTPEventTypes(AsgiTypeKeys):
    REQUEST = "http.request"
    RESPONSE_START = "http.response.start"
    RESPONSE_BODY = "http.response.body"
    DISCONNECT = "http.disconnect"


class WebSocketEvent(BaseEvent, total=False):
    subprotocol: str
    headers: RawHeaders
    bytes: Optional[bytes]
    text: Optional[str]
    code: int


class WebSocketEventTypes(AsgiTypeKeys):
    CONNECT = "websocket.connect"
    ACCEPT = "websocket.accept"
    RECEIVE = "websocket.receive"
    SEND = "websocket.send"
    DISCONNECT = "websocket.disconnect"
    CLOSE = "websocket.close"


class LifespanEvent(BaseEvent, total=False):
    message: str


class LifespanEventTypes(AsgiTypeKeys):
    STARTUP = "lifespan.startup"
    STARTUP_COMPLETE = "lifespan.startup.complete"
    STARTUP_FAILED = "lifespan.startup.failed"
    SHUTDOWN = "lifespan.shutdown"
    SHUTDOWN_COMPLETE = "lifespan.shutdown.complete"
    SHUTDOWN_FAILED = "lifespan.shutdown.failed"


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
    headers: RawHeaders
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
    headers: RawHeaders
    client: tuple[str, int]
    server: tuple[str, int]
    subprotocols: Iterable[str]


class LifespanScope(BaseScope):
    pass


"""
Event handler classes.
"""


class BaseEventHandler:
    """
    Base class for WS and HTTP event handlers.
    
    These classes abstract the handling of protocol messages.
    """
    def __init_subclass__(cls, *, events, states, **kwargs):
        """`events` should be subclasses of AsgiTypeKeys."""
        super().__init_subclass__(**kwargs)
        cls.events = events.typekeys()
        cls.states = states

    def __init__(self, scope, receive, send):
        self.scope = scope
        self._receive = receive
        self._send = send
        # Create a new instance of the state 
        self.state = self.states()

    def validate_event_type(self, event):
        if event["type"] not in self.events:
            raise ValueError(f"Unknown event type {event['type']} for {self.scope['type']}.")

    async def receive(self):
        raise NotImplementedError("Subclasses must override this method.")

    async def send(self, *args, **kwargs):
        raise NotImplementedError("Subclasses are required to override this method.")


class HTTPEventHandler(BaseEventHandler, events=HTTPEventTypes, states=HTTPConnectionState):
    """
    Handle HTTP event messages; manage state and .
    """
    async def receive(self):
        if self.state.finished:
            raise ValueError("Attempt to call a closed connection.")

        if not self.state.started:
            self.state.started = True

            while self.state.more_req_body:
                msg = await self._receive()

                if msg["type"] == "http.request":
                    self.state.body += msg["body"]
                    self.state.more_req_body = msg["more_body"]

            return {
                "type": "http.request",
                "body": self.state.req_body,
                "more_body": False,
            }

        return msg # Otherwise, the event is an http.disconnect event

    async def send(self, response: Union[Text, dict], finished: bool=False):
        """
        Response is either a dictionary of HTTP event keys or a text sequence.
        
        If a dict, the keys in the dict ar eused to update the connection state.
        If the dict has no key for 'body', the response is not sent. If it does,
        the 'http.response.start' message is sent.

        When finished is True, the connection state is updated to reflect the closing
        of the connection.
        """
        self.state.finished = finished
        self.state.more_body = finished

        if isinstance(response, dict):
            if "status" in response:
                self.state.status = response["status"]

            if "headers" in response:
                self.state.headers.update(response["headers"])

            if "body" in response:
                self.state.body += response["body"]
                self.state.resp_started = True

                event = {
                    "type": "http.response.start",
                    "status": self.state.status,
                    "headers": self.state.encode_headers(),
                }

                await self._send(event)

            return # wait to send the response start until the full response
                   # start has been received

        if isinstance(response, str):
            response = str.encode('utf8')

        self.state.body += response

        if self.state.finished:
            event = {
                "type": "http.response.body",
                "body": self.state.body,
                "more_body": False,
            }



class WebSocketEventHandler(BaseEventHandler, events=WebSocketEventTypes, states=WebSocketConnectionState):
    """
    Handle WebSocket event messages.
    """

    def validate_connection(self):
        return True

    async def receive(self):
        if self.state.finished:
            raise ValueError("Attempt to call a closed connection.")

        if not self.state.started:
            self.state.started = True

        msg = await self._receive()
        msgtype = msg["type"]

        if msgtype == "websocket.connect":
            if self.validate_connection():
                await self._send({
                    "type": "websocket.accept",
                    "subprotocol": self.state.subprotocol,
                    "headers": self.state.encode_headers(),
                })
                

            else:
                self.state.finished = True
                await self._send({
                    "type": "websocket.close",
                    "code": 1006,
                })

            return

        elif msgtype == "websocket.receive":
            if msg["bytes"] or msg["text"]:
                return msg

            else:
                raise ValueError("No content in message")

        elif msgtype == "websocket.disconnect":
            # return the disconnect event so the endpoint can do cleanup
            return msg

    async def send(self, txt: Optional[str]=None, bts: Optional[bytes]=None, *, close: bool=False, code: Optional[int]=None):
        """
        When finished is True, the connection state is updated to reflect the closing
        of the connection.
        """
        if self.state.finished and txt is not None or bts is not None:
            raise ValueError("Cannot send data over a closed connection.")

        if close:
            event = {
                "type": "websocket.disconnect",
                "code": code or self.status.code,
            }

            await self._send(event)

            self.state.finished = True

        elif txt is None and bts is None:
            raise ValueError("At least one of text and bytes is required.")

        else:
            event = {
                "type": "websocket.send",
                "text": txt,
                "bytes": bts,
            }

            await self._send(event)
            return
