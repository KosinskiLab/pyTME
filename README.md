# Python Template Matching Engine (PyTME)

A software for template matching on electron microscopy data.

📖 **[Official Documentation](https://kosinskilab.github.io/pyTME/)** | 🚀 **[Installation Guide](https://kosinskilab.github.io/pyTME/quickstart/installation.html)** | 📚 **[API Reference](https://kosinskilab.github.io/pyTME/reference/index.html)**

### Installation

There are three ways to get PyTME up and running:

1. **Install from Source:**
```bash
pip install git+https://github.com/KosinskiLab/pyTME
```

2. **Install from PyPi:**
```bash
pip install pytme
```

3. **Docker Container:**
Build and use the PyTME container from Docker. This assumes a linux/amd64 platform.
```bash
docker build \
	-t pytme \
	--platform linux/amd64 \
	-f docker/Dockerfile \
	.
```
🔗 For more on the Docker container, visit the [Docker Hub page](https://hub.docker.com).

---

### Publication

The pyTME publication is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.10.23.563472v1).

```
@article{Maurer2023.10.23.563472,
	author = {Valentin J. Maurer and Marc Siggel and Jan Kosinski},
	journal = {bioRxiv},
	title = {PyTME (Python Template Matching Engine): A fast, flexible, and multi-purpose template matching library for cryogenic electron microscopy data},
	year = {2023}}
```
