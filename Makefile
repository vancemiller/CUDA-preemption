SUBDIRS := $(wildcard */.)
SUBDIRS := $(filter-out common/., $(SUBDIRS))
CLEANDIRS := $(SUBDIRS:%=clean-%)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean: $(CLEANDIRS)

$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: all $(SUBDIRS)
.PHONY: clean $(CLEANDIRS)
