Theoretical Framework
=====================

.. contents::
   :depth: 2
   :local:

This page covers the core scientific and statistical concepts that underpin this toolkit. Understanding this framework is essential for correctly applying the model, interpreting your results, and understanding the assumptions being made.

For a practical application of these concepts, see the Full Walkthrough tutorial.

Basics of Calcium Imaging and Photometry
----------------------------------------

Calcium imaging is a cornerstone of modern systems neuroscience. It allows us to indirectly measure neuronal activity by tracking changes in intracellular calcium (Ca²⁺) concentration.

When a neuron fires an action potential, voltage-gated Ca²⁺ channels open, causing an influx of calcium into the cell.

We use fluorescent indicators (e.g., genetically encoded GCaMP or synthetic dyes) that bind to Ca²⁺ and increase their fluorescence.

By monitoring these changes in fluorescence with a microscope, we get a time-series signal that serves as a proxy for the underlying spiking activity of one or many neurons (Grienberger & Konnerth, 2012).

Fiber photometry is a related technique that uses an implanted optical fiber to measure the bulk fluorescence from a population of neurons in a specific brain region (Resendez et al., 2016).

Both methods generate time-series data where the signal's amplitude at any given time reflects the recent level of neural activity. This toolkit is designed to analyze these time-series signals.

Encoding Properties of Neurons
------------------------------

Neural encoding describes how neurons represent and transmit information about the external world or internal states. A neuron's encoding properties define the relationship between its activity and specific variables, such as sensory stimuli or behavioral actions (Dayan & Abbott, 2001).

For example, a neuron in the motor cortex might increase its activity just before an animal presses a lever. By modeling this relationship, we can characterize that neuron's encoding of "lever press preparation." This toolkit uses regression models to formally describe and quantify these encoding properties.

Linear Regression in Neuroscience
---------------------------------

Linear regression is a statistical method used to model the relationship between a dependent variable (our neural activity) and one or more independent, or predictor, variables (our behavioral events).

In its simplest form, the model assumes that the neural signal at time t can be described as a weighted sum of the behavioral events happening at that same time t:

.. code-block:: python

   NeuralActivity(t) ≈ β₀ + β₁ ⋅ EventA(t) + β₂ ⋅ EventB(t) + ... + ε

Here, the coefficients (β) represent the strength and direction of the relationship. A positive β for "Reward" would mean the neuron's activity tends to be higher when a reward is delivered. This toolkit uses regularized linear regression (Ridge) to find the optimal coefficients while preventing overfitting (Paninski et al., 2007).

The Time Kernel: Capturing Temporal Dynamics
--------------------------------------------

A simple linear regression is limited because it assumes the relationship between neural activity and a behavior is instantaneous. However, neural responses are not instant—they evolve over time. A neuron might fire before, during, and after a behavioral event.

To capture these rich temporal dynamics, we incorporate a time kernel into the regression model.

.. image:: images/time_kernel_diagram.png
   :alt: Time Kernel Diagram
   :align: center

Instead of estimating a single coefficient (β) for each behavioral event, we estimate a series of coefficients that represent the event's influence at different time lags. Crucially, while the model is still linear with respect to its parameters (the coefficients), this technique allows it to capture complex, non-linear dynamics over time. This gives us a powerful yet interpretable way to model intricate neural responses.

To estimate the kernel smoothly, the model represents its shape using a combination of simpler basis functions. Think of this as building a complex curve out of a set of simple, predefined 'Lego blocks', which ensures the final kernel is smooth and less noisy.

.. image:: images/full_model_concept.png
   :alt: Full Model Concept Diagram
   :align: center

Connecting Theory to Practice
----------------------------

In our toolkit, this entire process is managed by the ``fit_glm()`` function.

The predictor variables (like EventA and EventB in the equation) are the list of strings you pass to the ``regressors`` parameter.

The time kernel is defined by the ``time_kernel_window`` parameter. When you set ``time_kernel_window=(-1.0, 3.0)``, you are telling the model to learn the coefficients for the neuron's response from 1 second before to 3 seconds after each event.

See also:
See the guide on Interpreting Results for a detailed guide on how to read and understand kernel plots.

References
----------

Dayan, P., & Abbott, L. F. (2001). Theoretical neuroscience: Computational and mathematical modeling of neural systems. MIT Press.

Grienberger, C., & Konnerth, A. (2012). Imaging calcium in neurons. Neuron, 73(5), 862-885.

Paninski, L., Pillow, J., & Lewi, J. (2007). Statistical models for neural encoding, decoding, and optimal stimulus design. Progress in Brain Research, 165, 493-507.

Pillow, J. W., et al. (2008). Spatio-temporal correlations and visual signalling in a complete neuronal population. Nature, 454(7207), 995-999.

Resendez, S. L., et al. (2016). Visualization of cortical, subcortical and deep brain neural circuit dynamics during naturalistic mammalian behavior... Nature Protocols, 11(3), 566-597.
