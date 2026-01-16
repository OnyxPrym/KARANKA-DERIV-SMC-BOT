import multiprocessing

# Worker configuration
workers = 1  # For Render free tier
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 5

# Server socket
bind = '0.0.0.0:5000'
backlog = 2048

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'karanka-bot'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None

# Server hooks
def on_starting(server):
    pass

def on_reload(server):
    pass

def when_ready(server):
    pass

def pre_fork(server, worker):
    pass

def post_fork(server, worker):
    pass

def post_worker_init(worker):
    pass

def worker_int(worker):
    pass

def worker_abort(worker):
    pass

def pre_exec(server):
    pass

def pre_request(worker, req):
    pass

def post_request(worker, req, environ, resp):
    pass

def worker_exit(server, worker):
    pass

def nworkers_changed(server, new_value):
    pass

def on_exit(server):
    pass
