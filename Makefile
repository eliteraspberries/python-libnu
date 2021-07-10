.PHONY: test
test:
	py.test

.PHONY: cleanup
cleanup:
	rm -f *.pyc libnu/*.pyc

.PHONY: clean
clean: cleanup
