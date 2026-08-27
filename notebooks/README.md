# Tutorial — Histograms, weights, and Data/MC

This tutorial is intended to be the first hands-on exercise in the C2ST/DCTR repository.

## Recommended order

1. Complete Sections 2–5 using only toy arrays.
2. Load the real DY-VR tables in Section 6.
3. Compare raw MC row counts with weighted MC.
4. Produce before/after DY-correction Data/MC plots.
5. Compare physical- and shape-normalized views.
6. Inspect negative MC weights.
7. Complete the open-ended feature/process exercises.
8. Move on to the toy C2ST tutorial.

## Running the tutorial on NAF

The notebook should run on NAF, where the input files and the `c2st` environment are available, but it is usually much more convenient to display JupyterLab in the web browser of your local laptop.

This can be done with SSH port forwarding.

### 1. Connect to NAF

Open a terminal on your laptop and connect to NAF as usual:

```bash
ssh <your-naf-host>
```

Move to the repository:

```bash
cd HH_C2ST_DCTR
```

Activate the project environment:

```bash
source ~/miniforge3/bin/activate
conda activate c2st
```

### 2. Start JupyterLab on NAF

From the repository root, start JupyterLab without trying to open a browser on the remote machine:

```bash
jupyter lab --no-browser --port=8888
```

Jupyter will print a URL containing an authentication token, for example:

```text
http://localhost:8888/lab?token=...
```

Leave this terminal running while you work with the notebook.

### 3. Open an SSH tunnel from your laptop

Open a **second terminal on your laptop** and create a tunnel from local port `8888` to port `8888` on the NAF machine:

```bash
ssh -N -L 8888:localhost:8888 <your-naf-host>
```

Keep this terminal open as well.

The tunnel forwards

```text
your laptop:8888
        ↓
SSH connection
        ↓
NAF:8888
        ↓
JupyterLab
```

### 4. Open JupyterLab in your local browser

On your laptop, open:

```text
http://localhost:8888
```

If Jupyter asks for a token, copy the token printed by the `jupyter lab` command running on NAF.

You are now using the browser on your laptop, while all Python code, file access, and computations are still running on NAF inside the `c2st` environment.

### 5. If port 8888 is already in use

Choose another local port, for example `8890`:

```bash
ssh -N -L 8890:localhost:8888 <your-naf-host>
```

and then open:

```text
http://localhost:8890
```

The remote Jupyter server can still run on port `8888`; only the local port changes.

### 6. Open the tutorial notebook

From JupyterLab, navigate to:

```text
notebooks/
```

and open:

```text
01_histograms_weights_datamc.ipynb
```

The notebook should use the `c2st` Python kernel.

If needed, select it from the Jupyter kernel menu as:

```text
Python 3 (c2st)
```

This setup keeps the computation close to the NAF input files while giving you the much faster and more convenient browser interface on your local machine.