"""
Argument definitions for the DFT plugin.

This module defines a small Pydantic model that can be used by
Axolotl's configuration system to enable or disable the DFT plugin.
By convention, the plugin looks for a ``dft`` section in the YAML
configuration and reads its ``enabled`` flag.  If omitted, the plugin
defaults to enabled.

Example YAML snippet::

    plugins:
      - axolotl_dft_plugin.DFTPlugin

    dft:
      enabled: false

The ``enabled`` flag is optional; if omitted or set to ``true`` the
plugin will override the trainer's loss.  Set to ``false`` to disable
the plugin without removing it from the plugins list.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DFTArgs(BaseModel):
    """Configuration options for the DFT plugin.

    The only option currently supported is :attr:`enabled`, which
    controls whether the plugin replaces the default loss with DFT.
    """

    enabled: bool = Field(
        default=True,
        description=
        "Whether to enable the DFT loss.  When false the plugin does not "
        "override the trainer's loss."
    )
