# Vial
vial is a python ASGI framework inspired by bottle. vial aims to provide an intuitive, easy-to-use framework with few dependencies, but
powerful tools for extension. vial mimmics many features of bottle, but the most important thing that inspired me was the simplicity and
clarity of the bottle framework.

## Overview
vial provides a predominantly functional API for building web applications, with a small number of utility classes and the underlying
classes on which that API is built made available for extension. 

## API
The core parts of vial's application building API are exposed as decorators. The main such decorators are:

* route      - decorates functions that handle endpoints.
* middleware - decorates functions that provide ASGI middleware services.
* plugin     - decorates functions that provide additional services to endpoints (middleware applies to all calls).
* event      - decorates functions that respond to the ASGI lifespan protocol.
* command    - decorates functions that can be run as command-line scripts.

## Supporting classes and datastructures
vial can also make use of the following specialized classes.

* Settings   - based on pydantic settings, these classes can be used to declare global and module-specific configurations.
* Database   - an abstract wrapper for DBAPI compatible databases that can compile and execute Jinja2-based SQL templates.
* Queries    - a class that manages queries compiled from templates (cooperative with Database).
* Model      - pydantic models.
* Connection - a representation of an ASGI connection (holds the scope, receive, and send callables).
* Vial       - the base application class.

## Internals
TODO

## examples

@get("/member/{id:int}")
async def member(conn, id: int, /, email, username): # all parameters after the path parameter are query parameters
	m = await conn.db.execute("sql/members/@from_id", id=id) # the '@<query>' syntax specifies a query within a template file
	return 200, {"member": m} # vial automatically infers the return type from the python type, but it can also be specifically 
	                          # annotated with a return annotation (a serializable datastructure is considered a JSON response, 
							  # for example). Vial also allows a status code to be returned with the response. For complex cases,
							  # a Response object with additional explicit parameters can be returned.


@middleware("http")
async def db_session_middleware(conn, call_next):
	db = await conn.app.db.acquire()
	conn.db = db

	response = 500, "Internal server error"

	try:
		response = await call_next(conn)
		
	finally:
		await db.close()
		
	return response


@command("init-db", help="Initialize the database.")
@argument("dialect", choices={"sqlite", "mysql", "pgsql"}) # arguments and options can be defined with decorators
async def init_db(schema_path: str): # but vial can also automatically infer command-line arguments from the function signature.
	await_some_init_function(schema_path, dialect)
