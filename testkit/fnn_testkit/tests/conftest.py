# The plugin is also registered through the ``pytest11`` entry point when the
# package is installed; the entry point uses the same name so this is a no-op then.
pytest_plugins = ["fnn_testkit.plugin"]
