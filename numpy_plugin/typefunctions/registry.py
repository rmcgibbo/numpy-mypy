from functools import wraps

registry = {}


def register(name: str):	
	def inner(f):
		registry[name] = f
		return f
	return inner

