Theoretical Framework
=====================

This page covers the core scientific and statistical concepts that underpin this toolkit. Understanding this framework is essential for correctly applying the model, interpreting your results, and understanding the assumptions being made.

For a practical application of these concepts, see the :doc:`Full Walkthrough <walkthrough>`.

Contents
--------

- Basics of Calcium Imaging and Photometry
- Encoding Properties of Neurons
- Linear Regression in Neuroscience
- The Time Kernel: Capturing Temporal Dynamics
- Connecting Theory to Practice

Basics of Calcium Imaging and Photometry
----------------------------------------

Calcium imaging is a cornerstone of modern systems neuroscience. It allows us to indirectly measure neuronal activity by tracking changes in intracellular calcium (Ca²⁺) concentration.

- When a neuron fires an action potential, voltage-gated Ca²⁺ channels open, causing an influx of calcium into the cell.
- We use fluorescent indicators (e.g., genetically encoded GCaMP or synthetic dyes) that bind to Ca²⁺ and increase their fluorescence.
- By monitoring these changes in fluorescence with a microscope, we get a time-series signal that serves as a proxy for spiking activity.

Fiber photometry is a related technique that uses an implanted optical fiber to measure bulk fluorescence from a population of neurons in a brain region.

Both methods generate time-series data. This toolkit is designed to analyze these signals.

Encoding Properties of Neurons
------------------------------

Neural encoding describes how neurons represent information. A neuron's encoding properties define the relationship between its activity and variables like sensory stimuli or behavior.

Example:
- A motor cortex neuron might increase activity before an animal presses a lever.
- Modeling this helps us characterize how the neuron encodes "lever press preparation".

This toolkit uses regression models to quantify these encoding properties.

Linear Regression in Neuroscience
---------------------------------

Linear regression models the relationship between a dependent variable (neural activity) and independent variables (e.g., behavioral events).

.. code-block:: python

    NeuralActivity(t) ≈ β₀ + β₁ * EventA(t) + β₂ * EventB(t) + ... + ε

Where:
- `β₀, β₁, β₂` are coefficients describing the strength of each predictor
- Regularized linear regression (Ridge) helps prevent overfitting

The Time Kernel: Capturing Temporal Dynamics
--------------------------------------------

Simple regression assumes the effect of a behavior is instantaneous. In reality, neurons respond across time — before, during, and after a behavior.

To model this, we use a **time kernel**.

.. note::
    [INSERT DIAGRAM 1: Anatomy of a Time Kernel]
    A plot with:
    - x-axis: "Time Relative to Event (s)" from -1 to +2
    - y-axis: "Response Weight"
    - A peak after the event at t=0 and a decay afterward

Instead of one β per event, we estimate multiple βs across time lags. This still fits within a linear model but captures non-linear dynamics.

To estimate the kernel smoothly, the model uses **basis functions** — think of them as Lego blocks that build the full shape. This ensures smoother, less noisy estimates.

.. note::
    [INSERT DIAGRAM 2: Full Model Concept]
    Flowchart: 
    "Behavioral Events" + "Neural Activity" → Model → "Time Kernels" → "Predicted Activity"

Connecting Theory to Practice
-----------------------------

In this toolkit, the whole process is handled by the ``fit_glm()`` function.

Key parameters:
- ``regressors``: the predictor variable names (e.g., EventA, EventB)
- ``time_kernel_window``: the time span of the kernel (e.g., (-1.0, 3.0))

Example:
- ``time_kernel_window=(-1.0, 3.0)`` means we estimate how neural activity relates to events from 1 second before to 3 seconds after.

See also:
- :doc:`Interpreting Results <results>` for a guide on reading kernel plots.

References
----------

- Dayan, P., & Abbott, L. F. (2001). *Theoretical neuroscience*. MIT Press.
- Grienberger, C., & Konnerth, A. (2012). Imaging calcium in neurons. *Neuron*, 73(5), 862–885.
- Paninski, L., et al. (2007). Statistical models for neural encoding. *Progress in Brain Research*, 165, 493–507.
- Pillow, J. W., et al. (2008). Spatio-temporal correlations. *Nature*, 454(7207), 995–999.
- Resendez, S. L., et al. (2016). Visualization of neural dynamics. *Nature Protocols*, 11(3), 566–597.
