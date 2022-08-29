## A Simple Ray Tracer in JAX

<p align="center">
  <img src="outputs/rendered.png" alt="A sample render">
</p>

A JAX implementation of
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

### Instructions

Clone the repository:

```shell
git clone https://github.com/google-research/google-research.git
cd google-research/jaxraytrace
```

Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install dependencies with:

```shell
pip install --upgrade pip && pip install -r requirements.txt
```

Run the renderer with:

```shell
python main.py
```

Configure the scene by editing `configuration.py`.

## Authors

Ameya Daigavane