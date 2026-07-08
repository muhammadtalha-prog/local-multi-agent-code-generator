import ast
import sys
from utils.logger import logger

# Retrieve standard library modules from sys or fallback
try:
    KNOWN_STDLIB = set(sys.stdlib_module_names)
except AttributeError:
    # Robust fallback list for older Python versions
    KNOWN_STDLIB = {
        "abc", "argparse", "array", "ast", "asynchat", "asyncio", "asyncore", "atexit", 
        "base64", "bdb", "binascii", "bisect", "builtins", "bz2", "calendar", "cgi", 
        "cgitb", "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections", 
        "colorsys", "compileall", "concurrent", "configparser", "contextlib", "contextvars", 
        "copy", "copyreg", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime", 
        "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email", "encodings", 
        "ensurepip", "errno", "faulthandler", "filecmp", "fileinput", "fnmatch", "fractions", 
        "ftplib", "functools", "gc", "getopt", "getpass", "gettext", "glob", "graphlib", 
        "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http", "imaplib", "imghdr", 
        "imp", "importlib", "inspect", "io", "ipaddress", "itertools", "json", "keyword", 
        "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", 
        "math", "mimetypes", "mmap", "modulefinder", "multiprocessing", "netrc", "nis", 
        "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib", 
        "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib", 
        "poplib", "posix", "pprint", "profile", "pstats", "pty", "pwd", "py_compile", 
        "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline", "reprlib", 
        "resource", "rlcompleter", "runpy", "sched", "secrets", "select", "selectors", 
        "shelve", "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket", 
        "socketserver", "spwd", "sqlite3", "ssl", "stat", "statistics", "string", 
        "stringprep", "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig", 
        "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "test", 
        "textwrap", "threading", "time", "timeit", "tkinter", "token", "tokenize", 
        "trace", "traceback", "tracemalloc", "tty", "types", "typing", "unicodedata", 
        "unittest", "urllib", "uu", "uuid", "warnings", "wave", "weakref", "webbrowser", 
        "wsgiref", "xdrlib", "xml", "xmlrpc", "zipfile", "zipimport", "zlib", "zoneinfo"
    }

def check_imports(code):
    """
    Check the code for imports.
    Returns: (is_valid, error_message, external_packages)
    Always returns is_valid=True (unless syntax error which is already caught by the syntax checker).
    external_packages is a list of non-stdlib packages used.
    """
    logger.info("Running Static Import Checker Agent...")
    if not code:
        return True, None, []
        
    try:
        tree = ast.parse(code)
        external_packages = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split('.')[0]
                    if base not in KNOWN_STDLIB:
                        external_packages.add(base)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                base = node.module.split('.')[0]
                if base not in KNOWN_STDLIB:
                    external_packages.add(base)
        
        logger.info(f"Static Import Check passed. External packages used: {list(external_packages)}")
        return True, None, list(external_packages)
    except Exception as e:
        # Ignore syntax errors as they are validated by the syntax checker
        logger.debug(f"Static checker bypassed error during parsing (likely syntax error): {e}")
        return True, None, []


