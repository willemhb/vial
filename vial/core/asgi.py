from enum import Enum
import functools
from functools import cached_property
from http.cookies import SimpleCookie
from typing import Any, Awaitable, Callable, Optional, Union, Mapping
from urllib.parse import parse_qsl, unquote_plus
import json


from werkzeug.datastructures import Headers, MultiDictMap, MethodNotAllowed, NotFound, RequestRedirect, Rule

CoroutineFunction = Callable[[Any], Awaitable]


class ConnectionType(Enum):
    HTTP = "HTTP"
    WebSocket = "WebSocket"


class Connection:
    def __init__(
        self, scope: dict, *, send: CoroutineFunction, receive: CoroutineFunction
    ):
        self.scope = scope
        self.asgi_send = send
        self.asgi_receive = receive

        self.started = False
        self.finished = False
        self.resp_headers = Headers()
        self.resp_cookies: SimpleCookie = SimpleCookie()
        self.resp_status_code: Optional[int] = None

        self.http_body = b""
        self.http_has_more_body = True
        self.http_received_body_length = 0

    @cached_property
    def req_headers(self) -> Headers:
        headers = Headers()
        for (k, v) in self.scope["headers"]:
            headers.add(k.decode("ascii"), v.decode("ascii"))
        return headers

    @cached_property
    def req_cookies(self) -> SimpleCookie:
        cookie = SimpleCookie()
        cookie.load(self.req_headers.get("cookie", {}))
        return cookie

    @cached_property
    def type(self) -> ConnectionType:
        return (
            ConnectionType.WebSocket
            if self.scope.get("type") == "websocket"
            else ConnectionType.HTTP
        )

    @cached_property
    def method(self) -> str:
        return self.scope["method"]

    @cached_property
    def path(self) -> str:
        return self.scope["path"]

    @cached_property
    def query(self) -> MultiDict:
        return MultiDict(parse_qsl(unquote_plus(self.scope["query_string"].decode())))

    async def send(self, data: Union[bytes, str] = b"", finish: Optional[bool] = False):
        if self.finished:
            raise ValueError("No message can be sent when connection closed")
        if self.type == ConnectionType.HTTP:
            if isinstance(data, str):
                data = data.encode()
            await self._http_send(data, finish=finish)
        else:
            raise NotImplementedError()

    async def _http_send(self, data: bytes = b"", *, finish: bool = False):
        if not self.started:
            if finish:
                self.put_resp_header("content-length", str(len(data)))
            await self.start_resp()
        await self.asgi_send(
            {"type": "http.response.body", "body": data or b"", "more_body": True}
        )
        if finish:
            await self.finish()

    async def finish(self, close_code: Optional[int] = 1000):
        if self.type == ConnectionType.HTTP:
            if self.finished:
                raise ValueError("Connection already finished")
            if not self.started:
                self.resp_status_code = 204
                await self.start_resp()
            await self.asgi_send(
                {"type": "http.response.body", "body": b"", "more_body": False}
            )
        else:
            raise NotImplementedError()
            # await self.asgi_send({"type": "websocket.close", "code": close_code})
        self.finished = True

    async def start_resp(self):
        if self.started:
            raise ValueError("resp already started")
        if not self.resp_status_code:
            self.resp_status_code = 200
        headers = [
            [k.encode("ascii"), v.encode("ascii")] for k, v in self.resp_headers.items()
        ]
        for value in self.resp_cookies.values():
            headers.append([b"Set-Cookie", value.OutputString().encode("ascii")])
        await self.asgi_send(
            {
                "type": "http.response.start",
                "status": self.resp_status_code,
                "headers": headers,
            }
        )
        self.started = True

    async def body_iter(self):
        if self.type != ConnectionType.HTTP:
            raise ValueError("connection type is not HTTP")
        if self.http_received_body_length > 0 and self.http_has_more_body:
            raise ValueError("body iter is already started and is not finished")
        if self.http_received_body_length > 0 and not self.http_has_more_body:
            yield self.http_body
        req_body_length = (
            int(self.req_headers.get("content-length", "0"))
            if not self.req_headers.get("transfer-encoding") == "chunked"
            else None
        )
        while self.http_has_more_body:
            if req_body_length and self.http_received_body_length > req_body_length:
                raise ValueError("body is longer than declared")
            message = await self.asgi_receive()
            message_type = message.get("type")
            if message.get("type") == "http.disconnect":
                raise ValueError("Disconnected")
            if message_type != "http.request":
                continue
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                raise ValueError("Chunk is not bytes")
            self.http_body += chunk
            self.http_has_more_body = message.get("more_body", False) or False
            self.http_received_body_length += len(chunk)
            yield chunk

    async def body(self):
        return b"".join([chunks async for chunks in self.body_iter()])

    def put_resp_header(self, key, value):
        self.resp_headers.add(key, value)

    def put_resp_cookie(self, key, value):
        self.resp_cookies[key] = value



class HttpResponse:
    def __init__(
        self,
        body: Optional[Union[bytes, str]] = b"",
        connection: Optional[Connection] = None,
        *,
        status_code: int = 200,
        headers: Optional[Mapping[str, str]] = None
    ):
        self.body = body
        self.connection = connection
        self.status_code = status_code
        self.headers = headers

    def __await__(self):
        if not self.connection:
            raise ValueError("No connection")
        self.connection.resp_status_code = self.status_code
        if self.headers:
            for k, v in self.headers.items():
                self.connection.put_resp_header(k, v)
        return self.connection.send(self.body, finish=True).__await__()


class JsonResponse(HttpResponse):
    def __init__(
        self, data: Any, connection: Optional[Connection] = None, *args, **kwargs
    ):
        body = json.dumps(data)
        headers = kwargs.get("headers")
        if headers is None:
            headers = {}
        headers["content-type"] = "application/json"
        super().__init__(body, connection, *args, **kwargs)



class Router:
    def __init__(self):
        super().__init__()
        self.url_map = Map()
        self.endpoint_to_handler = {}

    def route(self, rule, methods=None, name=None):
        methods = set(methods) if methods is not None else None
        if methods and not "OPTIONS" in methods:
            methods.add("OPTIONS")

        def decorator(name: Optional[str], handler: Callable):
            self.add_route(
                rule_string=rule, handler=handler, methods=methods, name=name
            )
            return handler

        return functools.partial(decorator, name)

    def add_route(
        self,
        *,
        rule_string: str,
        handler: Callable,
        name: Optional[str] = None,
        methods: Optional[Iterable[str]] = None,
    ):
        if not name:
            name = handler.__name__
        existing_handler = self.endpoint_to_handler.get(name)
        if existing_handler and existing_handler is not handler:
            raise ValueError("Duplicated route name: %s" % (name))
        self.url_map.add(Rule(rule_string, endpoint=name, methods=methods))
        self.endpoint_to_handler[name] = handler

    def get_url_binding_for_connection(self, connection: Connection):
        scope = connection.scope
        return self.url_map.bind(
            connection.req_headers.get("host"),
            path_info=scope.get("path"),
            script_name=scope.get("root_path") or None,
            url_scheme=scope.get("scheme"),
            query_args=scope.get("query_string", b""),
        )

    async def __call__(self, connection: Connection):
        try:
            rule, args = self.get_url_binding_for_connection(connection).match(
                return_rule=True, method=connection.scope.get("method")
            )
        except RequestRedirect as e:
            connection.resp_status_code = 302
            connection.put_resp_header("location", e.new_url)
            return await connection.send(f"redirecting to: {e.new_url}", finish=True)
        except MethodNotAllowed:
            connection.resp_status_code = 405
            return await connection.send(b"", finish=True)
        except NotFound:
            pass
        else:
            handler = self.endpoint_to_handler[rule.endpoint]
            res = await handler(connection, **args)
            if isinstance(res, HttpResponse):
                res.connection = connection
                await res
