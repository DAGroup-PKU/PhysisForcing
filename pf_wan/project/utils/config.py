import os.path as osp
from typing import Any, Dict
import json

from jsonc_parser.parser import JsoncParser
from yacs.config import CfgNode as _CfgNode
from yacs.config import _assert_with_logging, _check_and_coerce_cfg_value_type


def load_json(filename):
    with open(filename, "r") as f:
        return JsoncParser.parse_str(f.read())


class CfgNode(_CfgNode):
    """
    New Features:
        1. use .jsonc instead of .yaml as configuration file
        2. support override_list for each CfgNode to override everything available
        3. support .jsonc path as value to CfgNode and automatically load it
        4. support __inherit__ .jsonc while default override it with other specified configs
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        """
        Args:
            init_dict (dict): the possibly-nested dictionary to initailize the CfgNode.
            key_list (list[str]): a list of names which index this CfgNode from the root.
                Currently only used for logging purposes.
            new_allowed (bool): whether adding new key is allowed when merging with
                other configs.
        """
        if init_dict:
            override_list = init_dict.pop("override_list", [])
        else:
            override_list = []

        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        init_dict = self._create_config_tree_from_dict(init_dict, key_list)
        super(CfgNode, self).__init__(init_dict)
        # Manage if the CfgNode is frozen or not
        self.__dict__[CfgNode.IMMUTABLE] = False
        # Deprecated options
        # If an option is removed from the code and you don't want to break existing
        # yaml configs, you can add the full config key as a string to the set below.
        self.__dict__[CfgNode.DEPRECATED_KEYS] = set()
        # Renamed options
        # If you rename a config option, record the mapping from the old name to the new
        # name in the dictionary below. Optionally, if the type also changed, you can
        # make the value a tuple that specifies first the renamed key and then
        # instructions for how to edit the config file.
        self.__dict__[CfgNode.RENAMED_KEYS] = {
            # 'EXAMPLE.OLD.KEY': 'EXAMPLE.NEW.KEY',  # Dummy example to follow
            # 'EXAMPLE.OLD.KEY': (                   # A more complex example to follow
            #     'EXAMPLE.NEW.KEY',
            #     "Also convert to a tuple, e.g., 'foo' -> ('foo',) or "
            #     + "'foo:bar' -> ('foo', 'bar')"
            # ),
        }

        # Allow new attributes after initialisation
        self.__dict__[CfgNode.NEW_ALLOWED] = new_allowed

        if len(override_list) > 0:
            self.merge_from_list(override_list)

    def init(self):
        assert not hasattr(self, "__defaults__"), "CfgNode has already been initialized!"
        self.__defaults__ = self.clone()
        self.__defaults__.freeze()
        for k, v in self.items():
            if k != "__defaults__" and isinstance(v, CfgNode):
                v.init()

    @classmethod
    def parse_config(cls, filename, cfg_node=None) -> Dict[str, Any]:
        def get_subnode(_cfg_node, key_or_index):
            if _cfg_node is not None:
                try:
                    return _cfg_node[key_or_index]
                except (KeyError, IndexError):
                    return None

        def recurse(entry, _cfg_node=None):
            if isinstance(entry, dict):
                result = {}
                if "__inherit__" in entry and isinstance(entry["__inherit__"], str) \
                    and osp.splitext(entry["__inherit__"])[1] in [".json", ".jsonc"]:
                    result.update(cls.parse_config(entry["__inherit__"], cfg_node=_cfg_node))
                    entry.pop("__inherit__")
                for k, v in entry.items():
                    result[k] = recurse(v, get_subnode(_cfg_node, k))
                return result
            elif isinstance(entry, list):
                return [recurse(x, _cfg_node=get_subnode(_cfg_node, i)) for i, x in enumerate(entry)]
            else:
                if isinstance(entry, str) and osp.splitext(entry)[1] in [".json", ".jsonc"]:
                    if isinstance(_cfg_node, CfgNode):
                        return cls.parse_config(entry, cfg_node=_cfg_node)
                    return entry
                else:
                    return entry

        return recurse(load_json(filename), _cfg_node=cfg_node)

    def merge_from_file(self, cfg_filename):
        """Load a jsonc config file and merge it this CfgNode."""
        assert osp.exists(cfg_filename), f"{cfg_filename} does not exist!"
        loaded_cfg = self.parse_config(cfg_filename, cfg_node=self)
        loaded_cfg = CfgNode(loaded_cfg, new_allowed=True)
        self.merge_from_other_cfg(loaded_cfg)

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    subkey in d, "Non-existent key: {}".format(full_key)
                )
                d = d[subkey]
            subkey = key_list[-1]
            # _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
            if subkey in d:
                if isinstance(d[subkey], CfgNode) and osp.splitext(v)[1] in [".json", ".jsonc"]:
                    new_config = d[subkey].__defaults__.clone()
                    new_config.defrost()
                    new_config.merge_from_file(v)
                    d[subkey] = new_config
                else:
                    value = self._decode_cfg_value(v)
                    value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
                    d[subkey] = value
            else:
                if isinstance(d, CfgNode):
                    value = self._decode_cfg_value(v)
                    d[subkey] = value

    def dump(self):
        def clean(_cfg_node):
            _cfg_node.pop("__defaults__", None)
            for v in _cfg_node.values():
                if isinstance(v, CfgNode):
                    clean(_cfg_node=v)

        # remove __defaults__ on dump
        clean(_cfg_node=self)
        return json.dumps(self, indent=2)


# global cfg
gcfg = CfgNode(new_allowed=True)
