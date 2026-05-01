"""Local Isaac Sim compatibility patches used by DreamBC scripts."""

from __future__ import annotations


def apply_camera_pipeline_patches() -> None:
    patch_omnigraph_attribute_helper_dtype()
    patch_syntheticdata_dependency_dtype()


def patch_omnigraph_attribute_helper_dtype() -> None:
    import numpy as np
    import omni.graph.core as og

    helper_cls = og.AttributeValueHelper
    if getattr(helper_cls, "_dreambc_unknown_dtype_patch", False):
        return

    original_set = helper_cls.set

    def _set_with_dtype_fallback(self, new_value, on_gpu=False, update_usd=False):
        attribute_name = None
        try:
            attribute_name = self._AttributeValueHelper__attribute.get_name()
        except Exception:
            pass
        try:
            return original_set(self, new_value, on_gpu=on_gpu, update_usd=update_usd)
        except TypeError as exc:
            if "unknown dtype" not in str(exc):
                raise
            if attribute_name == "inputs:simTimesToWrite":
                try:
                    coerced = np.asarray(new_value, dtype=np.float64).reshape(-1)
                    return original_set(self, coerced, on_gpu=on_gpu, update_usd=update_usd)
                except Exception:
                    try:
                        seq = [float(v) for v in list(new_value)]
                        coerced = np.asarray(seq, dtype=np.float64).reshape(-1)
                        return original_set(self, coerced, on_gpu=on_gpu, update_usd=update_usd)
                    except Exception:
                        if not getattr(helper_cls, "_dreambc_sim_times_warning_printed", False):
                            print(
                                "Warning: could not coerce Replicator inputs:simTimesToWrite "
                                f"payload; skipping write. Original error: {exc}"
                            )
                            helper_cls._dreambc_sim_times_warning_printed = True
                        return None
            if not isinstance(new_value, (list, tuple, np.ndarray)):
                raise
            for dtype in (np.float64, np.uint64, np.int64):
                try:
                    return original_set(
                        self,
                        np.asarray(new_value, dtype=dtype),
                        on_gpu=on_gpu,
                        update_usd=update_usd,
                    )
                except TypeError:
                    continue
            raise

    helper_cls.set = _set_with_dtype_fallback
    helper_cls._dreambc_unknown_dtype_patch = True
    print("Applied DreamBC OmniGraph AttributeValueHelper dtype patch.")


def patch_syntheticdata_dependency_dtype() -> None:
    import numpy as np
    import omni.graph.core as og
    import omni.graph.tools.ogn
    from omni.syntheticdata import SyntheticData, SyntheticDataException
    from pxr import Vt

    if getattr(SyntheticData, "_dreambc_uint64_dependency_patch", False):
        return

    def _add_node_downstream_intergraph_dependency(node, downstream_node_handle) -> int:
        if (not node.is_valid()) or (downstream_node_handle is None):
            return 0
        dependency_attribute_name = "state:_sdp_intergraph_downstream_node_handles_"
        if not node.get_attribute_exists(dependency_attribute_name):
            dep_attrib = og.Controller.create_attribute(
                node=node,
                attr_name=dependency_attribute_name,
                attr_type=og.Type(og.BaseDataType.UINT64, 1, 1),
                attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE,
                attr_extended_type=og.ExtendedAttributeType.REGULAR,
                undoable=False,
            )
            if (dep_attrib is None) or not node.get_attribute_exists(dependency_attribute_name):
                raise SyntheticDataException(
                    f"failed to create node dependency for {node.get_prim_path()} @ {downstream_node_handle}."
                )
            dep_attrib.set_metadata(omni.graph.tools.ogn.MetadataKeys.INTERNAL, "1")
            dep_attrib.set_metadata(omni.graph.tools.ogn.MetadataKeys.LITERAL_ONLY, "1")
        else:
            dep_attrib = node.get_attribute(dependency_attribute_name)
        dep_attrib_data = dep_attrib.get_attribute_data()
        current = np.asarray(dep_attrib_data.get(), dtype=np.uint64).reshape(-1)
        dep_data = [int(value) for value in current] + [int(downstream_node_handle)]
        try:
            og.AttributeValueHelper(dep_attrib).set(Vt.UInt64Array(dep_data), update_usd=True)
        except TypeError as exc:
            try:
                dep_attrib_data.set(Vt.UInt64Array(dep_data))
            except TypeError:
                print(
                    "Warning: skipped SyntheticData intergraph dependency write "
                    f"for {node.get_prim_path()} -> {downstream_node_handle}: {exc}"
                )
        return len(dep_data)

    SyntheticData._add_node_downstream_intergraph_dependency = staticmethod(
        _add_node_downstream_intergraph_dependency
    )
    SyntheticData._dreambc_uint64_dependency_patch = True
    print("Applied DreamBC SyntheticData uint64 dependency patch.")

