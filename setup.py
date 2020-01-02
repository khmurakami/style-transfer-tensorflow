import setuptools

setuptools.setup(
    name="style_transfer",
    version="0.0.3",
    author="Kalani Murakami",
    author_email="kalanimurakami1218@gmail.com",
    description="Quick Style Transfer",
    packages=['style_transfer'],
    install_requires=["tensorflow-gpu", "tensorflow_hub", "Pillow"],
    license="MIT",
    url="https://github.com/khmurakami/style-transfer-tensorflow.git"
)
