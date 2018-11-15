import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

try:
    from setuptools_rust import RustExtension
except ImportError:
    import subprocess

    errno = subprocess.call([sys.executable, "-m", "pip", "install", "setuptools-rust"])
    if errno:
        print("Please install setuptools-rust package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension


class PyTest(TestCommand):
    user_options = []

    def run(self):
        self.run_command("test_rust")

        import subprocess

        subprocess.check_call(["pytest", "tests"])


setup_requires = ["setuptools-rust", "wheel"]
install_requires = []
tests_require = install_requires + ["pytest"]

setup(
    name="rs-embed",
    version="0.1.0",
    classifiers=[],
    packages=["rs_embed"],
    rust_extensions=[RustExtension("rs_embed.rs_embed", "Cargo.toml")],
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    include_package_data=True,
    zip_safe=False,
    cmdclass=dict(test=PyTest),
)
