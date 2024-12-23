Utils
=====

AutoUMAP
--------

__init__
~~~~~~~~
.. automethod:: solo.utils.auto_umap.AutoUMAP.__init__
   :noindex:

add_auto_umap_args
~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.auto_umap.AutoUMAP.add_auto_umap_args
   :noindex:

on_train_start
~~~~~~~~~~~~~~
.. automethod:: solo.utils.auto_umap.AutoUMAP.on_train_start
   :noindex:

plot
~~~~
.. automethod:: solo.utils.auto_umap.AutoUMAP.plot
   :noindex:

on_validation_end
~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.auto_umap.AutoUMAP.on_validation_end
   :noindex:


OfflineUMAP
-----------

__init__
~~~~~~~~
.. automethod:: solo.utils.auto_umap.OfflineUMAP.__init__
   :noindex:

plot
~~~~
.. automethod:: solo.utils.auto_umap.OfflineUMAP.plot
   :noindex:



Checkpointer
------------

__init__
~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.__init__
   :noindex:

add_checkpointer_args
~~~~~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.add_checkpointer_args
   :noindex:

initial_setup
~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.initial_setup
   :noindex:

save_args
~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.save_args
   :noindex:

save
~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.save
   :noindex:

on_train_start
~~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.on_train_start
   :noindex:

on_validation_end
~~~~~~~~~~~~~~~~~
.. automethod:: solo.utils.checkpointer.Checkpointer.on_validation_end
   :noindex:


Gather layer
------------

.. autofunction:: solo.utils.misc.gather
   :noindex:


Weighted KNN Classifier
-----------------------

__init__
~~~~~~~~
.. automethod:: solo.utils.knn.WeightedKNNClassifier.__init__
   :noindex:


update
~~~~~~
.. automethod:: solo.utils.knn.WeightedKNNClassifier.update
   :noindex:


compute
~~~~~~~
.. automethod:: solo.utils.knn.WeightedKNNClassifier.compute
   :noindex:



LARS
----

.. automethod:: solo.utils.lars.LARS.__init__
   :noindex:

Metrics
-------

accuracy_at_k
~~~~~~~~~~~~~
.. autofunction:: solo.utils.metrics.accuracy_at_k
   :noindex:

weighted_mean
~~~~~~~~~~~~~
.. autofunction:: solo.utils.metrics.weighted_mean
   :noindex:


Momentum module
---------------

MomentumUpdater
~~~~~~~~~~~~~~~
.. automethod:: solo.utils.momentum.MomentumUpdater.__init__
   :noindex:

initialize_momentum_params
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: solo.utils.momentum.initialize_momentum_params
   :noindex:


Sinkhorn-Knopp
--------------

.. automethod:: solo.utils.sinkhorn_knopp.SinkhornKnopp.__init__
   :noindex:

Whitening
---------

.. automethod:: solo.utils.whitening.Whitening2d.__init__
   :noindex:


PositionalEncoding1D
---------------------
:class:`PositionalEncoding1D` applies positional encoding to the last dimension of a 3D tensor.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncoding1D.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncoding1D.forward
   :noindex:

PositionalEncodingPermute1D
---------------------------
:class:`PositionalEncodingPermute1D` permutes the input tensor and applies 1D positional encoding.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncodingPermute1D.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncodingPermute1D.forward
   :noindex:

PositionalEncoding2D
---------------------
:class:`PositionalEncoding2D` applies positional encoding to the last two dimensions of a 4D tensor.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncoding2D.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncoding2D.forward
   :noindex:

PositionalEncodingPermute2D
---------------------------
:class:`PositionalEncodingPermute2D` permutes the input tensor and applies 2D positional encoding.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncodingPermute2D.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncodingPermute2D.forward
   :noindex:

PositionalEncoding3D
---------------------
:class:`PositionalEncoding3D` applies positional encoding to the last three dimensions of a 5D tensor.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncoding3D.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncoding3D.forward
   :noindex:

PositionalEncodingPermute3D
---------------------------
:class:`PositionalEncodingPermute3D` permutes the input tensor and applies 3D positional encoding.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncodingPermute3D.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.PositionalEncodingPermute3D.forward
   :noindex:

Summer
------
:class:`Summer` adds positional encoding to the original tensor.

__init__
~~~~~~~~
.. automethod:: solo.utils.positional_encoding.Summer.__init__
   :noindex:

forward
~~~~~~~
.. automethod:: solo.utils.positional_encoding.Summer.forward
   :noindex:
