Developer Guide
===============

Contributions are welcome. This page describes how to set up a development
environment and run the project's checks.

Environment
-----------

.. code-block:: bash

    conda env create -f devtools/conda-envs/test_env.yaml -n openfit_dev
    conda activate openfit_dev
    pip install -e . --no-deps

The test environment includes ``numba``, ``openmm``, ``mrcfile``, ``mdtraj`` and
``matplotlib`` so the full (non-skipped) test suite runs locally.

Running the tests
-----------------

.. code-block:: bash

    pytest openfit/tests/ -v
    pytest openfit/tests/ --cov=openfit --cov-report=term-missing

Tests that need optional dependencies (OpenMM, mrcfile, MolScene) are guarded
with ``pytest.importorskip`` and skip cleanly when those packages are absent.

Linting and formatting
----------------------

The project uses ``black`` (line length 119) for formatting and ``flake8`` for
linting; both run in CI and as pre-commit hooks.

.. code-block:: bash

    black openfit/
    flake8 openfit/

    pre-commit install        # enable hooks on every commit
    pre-commit run --all-files

Building the documentation
--------------------------

.. code-block:: bash

    pip install pydata-sphinx-theme sphinx-design sphinx-copybutton
    cd docs
    make html
    # open _build/html/index.html

Continuous integration
----------------------

``.github/workflows/CI.yaml`` runs a lint job (flake8 + black) and a test matrix
across operating systems and Python versions, uploading coverage to Codecov.
``codeql.yaml`` runs CodeQL security analysis.
