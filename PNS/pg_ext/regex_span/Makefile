EXTENSION = regex_span
MODULE_big = regex_span
OBJS = regex_span.o
DATA = regex_span--1.0.sql
DOCS = README.md
PGFILEDESC = "regex_span - character-based regex span helpers"

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
