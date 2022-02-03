# pylint: disable=C0302
"""
    Module for retrieving and writing data in H5parm format
    SPDX-License-Identifier: BSD-3-Clause
"""

import datetime
import itertools
import os
import re
import sys

import numpy as np
import tables

from ska_sdp_screen_fitting import _version
from ska_sdp_screen_fitting.lofar.lib_losoto import deprecated_alias
from ska_sdp_screen_fitting.utils._logging import logger as logging

if sys.version_info > (3, 0):
    from itertools import zip_longest
else:
    from itertools import izip_longest as zip_longest

# check for tables version
if int(tables.__version__.split(".", maxsplit=1)[0]) < 3:
    logging.critical(
        "pyTables version must be >= 3.0.0, found: %s", tables.__version__
    )
    sys.exit(1)


def open_soltab(
    h_5_parm_file,
    solset_name=None,
    soltab_name=None,
    address=None,
    readonly=True,
):
    """
    Convenience function to get a soltab object from an H5parm file and an
    address like "solset000/phase000".

    Parameters
    ----------
    h_5_parm_file : str
        H5parm filename.
    solset_name : str
        solset name
    soltab_name : str
        soltab name
    address : str
        solset/soltab name (to use in place of the parameters solset and
        soltab)
    readonly : bool, optional
        if True the table is open in readonly mode, by default True.

    Returns
    -------
    Soltab obj
        A solution table object.
    """
    h5_ = H5parm(h_5_parm_file, readonly)
    if solset_name is None or soltab_name is None:
        if address is None:
            logging.error(
                "Address must be specified if solset_name and soltab_name are"
                " not given."
            )
            sys.exit(1)
        solset_name, soltab_name = address.split("/")
    solset = h5_.get_solset(solset_name)
    return solset.get_soltab(soltab_name)


class H5parm:
    """
    Create an H5parm object.

    Parameters
    ----------
    h_5_parm_file : str
        H5parm filename.
    readonly : bool, optional
        if True the table is open in readonly mode, by default True.
    complevel : int, optional
        compression level from 0 to 9 when creating the file, by default 5.
    complib : str, optional
        library for compression: lzo, zlib, bzip2, by default zlib.
    """

    def __init__(
        self, h_5_parm_file, readonly=True, complevel=0, complib="zlib"
    ):

        self.table = None  # variable to store the pytable object
        self.filename = h_5_parm_file

        if os.path.isfile(h_5_parm_file):
            if not tables.is_hdf5_file(h_5_parm_file):
                logging.critical("Not a HDF5 file: %s.", h_5_parm_file)
                raise Exception(f"Not a HDF5 file: {h_5_parm_file}.")
            if readonly:
                logging.debug("Reading from %s.", h_5_parm_file)
                self.table = tables.open_file(
                    h_5_parm_file,
                    "r",
                    IO_BUFFER_SIZE=1024 * 1024 * 10,
                    BUFFER_TIMES=500,
                )
            else:
                logging.debug("Appending to %s.", h_5_parm_file)
                self.table = tables.open_file(
                    h_5_parm_file,
                    "r+",
                    IO_BUFFER_SIZE=1024 * 1024 * 10,
                    BUFFER_TIMES=500,
                )

            # Check if it's a valid H5parm file: attribute h5parm_version
            # should be defined in any solset
            is_h5parm = True
            for node in self.table.root:
                if "h5parm_version" not in node._v_attrs:
                    is_h5parm = False
                    break
            if not is_h5parm:
                logging.warning(
                    "Missing H5pram version. Is this a properly made H5parm?"
                )

        else:
            if readonly:
                raise Exception(f"Missing file {h_5_parm_file}.")
            logging.debug("Creating %s.", h_5_parm_file)
            # add a compression filter
            filter = tables.Filters(complevel=complevel, complib=complib)
            self.table = tables.open_file(
                h_5_parm_file,
                filters=filter,
                mode="w",
                IO_BUFFER_SIZE=1024 * 1024 * 10,
                BUFFER_TIMES=500,
            )

    def close(self):
        """
        Close the open table.
        """
        logging.debug("Closing table.")
        self.table.close()

    def __str__(self):
        """
        Returns
        -------
        string
            Info about H5parm contents.
        """
        return self.print_info()

    def make_solset(self, solset_name=None):
        """
        Create a new solset, if the provided name is not given or exists
        then it falls back on the first available sol###.

        Parameters
        ----------
        solset : str
            Name of the solution set.

        Returns
        -------
        solset obj
            Newly created solset object.
        """

        if isinstance(solset_name, str) and not re.match(
            r"^[A-Za-z0-9_-]+$", solset_name
        ):
            logging.warning(
                "Solution-set %s contains unsuported characters. "
                "Use [A-Za-z0-9_-]. Switching to default.",
                solset_name,
            )
            solset_name = None

        if solset_name in self.get_solset_names():
            logging.warning(
                "Solution-set %s already present. Switching to default.",
                solset_name,
            )
            solset_name = None

        if solset_name is None:
            solset_name = self._first_avail_solset_name()

        logging.info("Creating a new solution-set: %s.", solset_name)
        solset = self.table.create_group("/", solset_name)
        solset._f_setattr("h5parm_version", _version.__h5parmVersion__)

        return Solset(solset)

    def get_solsets(self):
        """
        Get all solution set objects.

        Returns
        -------
        list
            A list of all solsets objects.
        """
        return [
            Solset(solset) for solset in self.table.root._v_groups.values()
        ]

    def get_solset_names(self):
        """
        Get all solution set names.

        Returns
        -------
        list
            A list of str of all solsets names.
        """

        return [
            solset_name
            for solset_name in iter(list(self.table.root._v_groups.keys()))
        ]

    def get_solset(self, solset):
        """
        Get a solution set with a given name.

        Parameters
        ----------
        solset : str
            Name of the solution set.

        Returns
        -------
        solset obj
            Return solset object.
        """
        if solset not in self.get_solset_names():
            logging.critical("Cannot find solset: %s.", solset)
            raise Exception(f"Cannot find solset: {solset}.")

        return Solset(self.table.get_node("/", solset))

    def _first_avail_solset_name(self):
        """
        Find the first available solset name which has the form of "sol###".

        Returns
        -------
        str
            Solset name.
        """
        nums = []
        for solset_name in self.get_solset_names():
            if re.match(r"^sol[0-9][0-9][0-9]$", solset_name):
                nums.append(int(solset_name[-3:]))
        first_solset_idx = min(list(set(range(1000)) - set(nums)))
        return f"sol{first_solset_idx:03d}"

    def print_info(self, filter=None, verbose=False):
        """
        Used to get readable information on the H5parm file.

        Parameters
        ----------
        filter: str, optional
            Solution set name to get info for
        verbose: bool, optional
            If True, return additional info on axes

        Returns
        -------
        str
            Returns a string with info about H5parm contents.
        """

        def grouper(n_interables, iterable, fillvalue=" "):
            """
            Groups iterables into specified groups

            Parameters
            ----------
            n_interables : int
                number of iterables to group
            iterable : iterable
                iterable to group
            fillvalue : str
                value to use when to fill blanks in output groups

            Example
            -------
            grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
            """
            args = [iter(iterable)] * n_interables
            return zip_longest(fillvalue=fillvalue, *args)

        def wrap(text, width=80):
            """
            Wraps text to given width and returns list of lines
            """
            lines = []
            for paragraph in text.split("\n"):
                line = []
                len_line = 0
                for word in paragraph.split(" "):
                    word.strip()
                    len_word = len(word)
                    if len_line + len_word <= width:
                        line.append(word)
                        len_line += len_word + 1
                    else:
                        lines.append(" ".join(line))
                        line = [21 * " " + word]
                        len_line = len_word + 22
                lines.append(" ".join(line))
            return lines

        info = f"\nSummary of {self.filename}\n"
        solsets = self.get_solsets()

        # Filter on solset name
        if filter is not None:
            info += (
                f"\nFiltering on solution set name with filter = '{filter}'\n"
            )
            solsets = [
                solset for solset in solsets if re.search(filter, solset.name)
            ]

        if len(solsets) == 0:
            info += "\nNo solution sets found.\n"
            return info

        # delete axes value file if already present
        if verbose and os.path.exists(self.filename + "-axes_values.txt"):
            logging.warning("Overwriting %s-axes_values.txt", self.filename)
            os.system("rm " + self.filename + "-axes_values.txt")

        # For each solution set, list solution tables, sources, and antennas
        for solset in solsets:
            info += f"\nSolution set '{solset.name}':\n"
            info += "=" * len(solset.name) + "=" * 16 + "\n\n"

            # Add direction (source) names
            sources = sorted(solset.get_source().keys())
            info += "Directions: "
            for src_name1, src_name2, src_name3 in grouper(3, sources):
                info += f"{src_name1:}\t{src_name2:}\t{src_name3:}\n"

            # Add station names
            antennas = sorted(solset.get_ant().keys())
            info += "\nStations: "
            for ant1, ant2, ant3, ant4 in grouper(4, antennas):
                info += f"{ant1:}\t{ant2:}\t{ant3:}\t{ant4:}\n"

            # For each table, add length of each axis and history of
            # operations applied to the table.
            if verbose:
                logging.warning(
                    "Axes values saved in %s-axes_values.txt", self.filename
                )
                file = open(  # pylint: disable=R1732
                    self.filename + "-axes_values.txt", "a", encoding="utf8"
                )
            soltabs = solset.get_soltabs()
            names = np.array([s.name for s in soltabs])
            sorted_soltabs = [x for _, x in sorted(zip(names, soltabs))]
            for soltab in sorted_soltabs:
                try:
                    if verbose:
                        file.write(
                            "### /" + solset.name + "/" + soltab.name + "\n"
                        )
                    logging.debug("Fetching info for %s.", soltab.name)
                    axis_names = soltab.get_axes_names()
                    axis_str_list = []
                    for axis_name in axis_names:
                        nslots = soltab.get_axis_len(axis_name)
                        if nslots > 1:
                            pls = "s"
                        else:
                            pls = ""
                        axis_str_list.append(f"{nslots} {axis_name}{pls}")
                        if verbose:
                            file.write(axis_name + ": ")
                            vals = soltab.get_axis_values(axis_name)
                            # ugly hardcoded workaround to print all the
                            # important decimal values for time/freq
                            if axis_name == "freq":
                                file.write(
                                    " ".join([f"{v:.8f}" for v in vals])
                                    + "\n\n"
                                )
                            elif axis_name == "time":
                                file.write(
                                    " ".join([f"{v:.7f}" for v in vals])
                                    + "\n\n"
                                )
                            else:
                                file.write(
                                    " ".join([f"{v}" for v in vals]) + "\n\n"
                                )
                    axis_list_str = ", ".join(axis_str_list)
                    info += (
                        f"\nSolution table '{soltab.name}' (type: "
                        f"{soltab.get_type()}): {axis_list_str}\n"
                    )
                    weights = soltab.get_values(
                        weight=True, ret_axes_vals=False
                    )
                    vals = soltab.get_values(weight=False, ret_axes_vals=False)
                    flagged_data_perc = (
                        100.0
                        * np.sum(weights == 0 | np.isnan(vals))
                        / len(weights.flat)
                    )
                    info += f"    Flagged data: {flagged_data_perc:.3f}%\n"

                    # Add some extra attributes stored in screen-type tables
                    if soltab.get_type() == "screen":
                        attr_names = soltab.obj._v_attrs._v_attrnames
                        add_head = True
                        for name in attr_names:
                            if name in ["beta", "freq", "height", "order"]:
                                if add_head:
                                    info += "    Screen attributes:\n"
                                    add_head = False
                                info += (
                                    f"        {name}: "
                                    f"{soltab.obj._v_attrs[name]}\n"
                                )

                    # Add history
                    history = soltab.get_history()
                    if history != "":
                        info += 4 * " " + "History: "
                        joinstr = "\n" + 13 * " "
                        info += joinstr.join(wrap(history)) + "\n"
                except tables.exceptions.NoSuchNodeError:
                    info += (
                        f"\nSolution table '{soltab.name}': "
                        "No valid data found\n"
                    )

            if verbose:
                file.close()
        return info


class Solset:
    """
    Create a solset object

    Parameters
    ----------
    solset : pytables group
        The solution set pytables group object
    """

    def __init__(self, solset):

        if not isinstance(solset, tables.Group):
            logging.error(
                "Object must be initialized with a pyTables Group object."
            )
            sys.exit(1)

        self.obj = solset
        self.name = solset._v_name

    def close(self):
        """
        Close solset
        """
        self.obj._g_flushGroup()

    def delete(self):
        """
        Delete this solset.
        """
        logging.info('Solset "%s" deleted.', self.name)
        self.obj._f_remove(recursive=True)

    def rename(self, newname, overwrite=False):
        """
        Rename this solset.

        Parameters
        ----------
        newname : str
            New solution set name.
        overwrite : bool, optional
            Overwrite existing solset with same name.
        """
        self.obj._f_rename(newname, overwrite)
        logging.info('Solset "%s" renamed to "%s".', self.name, newname)
        self.name = self.obj._v_name

    def make_soltab(
        self,
        soltype=None,
        soltab_name=None,
        axes_names=[],
        axes_vals=[],
        vals=None,
        weights=None,
        parmdb_type="",
        weight_dtype="f16",
    ):
        """
        Create a Soltab into this solset.

        Parameters
        ----------
        soltype : str
            Solution-type (e.g. amplitude, phase)
        soltab_name : str, optional
            The solution-table name, if not specified is generated from the
            solution-type
        axes_names : list
            List with the axes names
        axes_vals : list
            List with the axes values (each is a separate np.array)
        chunk_shape : list, optional
            List with the chunk shape
        vals : numpy array
            Array with shape given by the axes_vals lenghts
        weights : numpy array
            Same shape of the vals array
            0->FLAGGED, 1->MAX_WEIGHT
        parmdb_type : str
            Original parmdb solution type
        weight_dtype : str
            THe dtype of weights allowed values are ('f16' or 'f32' or 'f64')

        Returns
        -------
        soltab obj
            Newly created soltab object
        """

        if soltype is None:
            raise Exception(
                "Solution-type not specified while adding a solution-table."
            )

        # checks on the soltab
        if isinstance(soltab_name, str) and not re.match(
            r"^[A-Za-z0-9_-]+$", soltab_name
        ):
            logging.warning(
                "Solution-table %s contains unsuported characters. "
                "Use [A-Za-z0-9_-]. Switching to default.",
                soltab_name,
            )
            soltab_name = None

        if soltab_name in self.get_soltab_names():
            logging.warning(
                "Solution-table %s already present. Switching to default.",
                soltab_name,
            )
            soltab_name = None

        if soltab_name is None:
            soltab_name = self._first_avail_soltab_name(soltype)

        logging.info("Creating a new solution-table: %s.", soltab_name)

        # check input
        assert len(axes_names) == len(axes_vals)
        dim = []
        for i, axis_name in enumerate(axes_names):
            dim.append(len(axes_vals[i]))
        assert dim == list(vals.shape)
        assert dim == list(weights.shape)

        # if input is OK, create table
        soltab = self.obj._v_file.create_group(
            "/" + self.name, soltab_name, title=soltype
        )
        soltab._v_attrs["parmdb_type"] = parmdb_type
        for i, axis_name in enumerate(axes_names):
            self.obj._v_file.create_array(
                "/" + self.name + "/" + soltab_name,
                axis_name,
                obj=np.array(axes_vals[i]),
            )

        # create the val/weight Carrays
        # val = self.obj._v_file.create_carray('/'+self.name+'/'+soltab_name,
        # 'val', obj=vals.astype(np.float64), chunk_shape=None,
        # atom=tables.Float64Atom())

        # weight = self.obj._v_file.create_carray('/'+self.name+'/'+
        # soltab_name,'weight', obj=weights.astype(np.float16),
        # chunk_shape=None, atom=tables.Float16Atom())

        # array do not have compression but are much faster
        val = self.obj._v_file.create_array(
            "/" + self.name + "/" + soltab_name,
            "val",
            obj=vals.astype(np.float64),
            atom=tables.Float64Atom(),
        )
        assert weight_dtype in [
            "f16",
            "f32",
            "f64",
        ], "Allowed weight dtypes are 'f16','f32', 'f64'"
        if weight_dtype == "f16":
            np_d = np.float16
            pt_d = tables.Float16Atom()
        elif weight_dtype == "f32":
            np_d = np.float32
            pt_d = tables.Float32Atom()
        elif weight_dtype == "f64":
            np_d = np.float64
            pt_d = tables.Float64Atom()
        weight = self.obj._v_file.create_array(
            "/" + self.name + "/" + soltab_name,
            "weight",
            obj=weights.astype(np_d),
            atom=pt_d,
        )
        axis_names = ",".join([axis_name for axis_name in axes_names])
        val.attrs["AXES"] = axis_names.encode()
        weight.attrs["AXES"] = axis_names.encode()

        return Soltab(soltab)

    def _first_avail_soltab_name(self, soltype):
        """
        Find the first available soltab name which
        has the form of "soltypeName###"

        Parameters
        ----------
        soltype : str
            Type of solution (amplitude, phase, RM, clock...)

        Returns
        -------
        str
            First available soltab name
        """
        nums = []
        for soltab in self.get_soltabs():
            if re.match(r"^" + soltype + "[0-9][0-9][0-9]$", soltab.name):
                nums.append(int(soltab.name[-3:]))
        soltype_idx = min(list(set(range(1000)) - set(nums)))
        return f"{soltype}{soltype_idx:03d}"

    def get_soltabs(self, use_cache=False, sel={}):
        """
        Get all Soltabs in this Solset.

        Parameters
        ----------
        use_cache : bool, optional
            soltabs obj will use cache, by default False
        sel : dict, optional
            selection dict, by default no selection

        Returns
        -------
        list
            List of solution tables objects for all available soltabs in this
            solset
        """
        soltabs = []
        for soltab in self.obj._v_groups.values():
            soltabs.append(Soltab(soltab, use_cache, sel))
        return soltabs

    def get_soltab_names(self):
        """
        Get all Soltab names in this Solset.

        Returns
        -------
        list
            List of str for all available soltabs in this solset.
        """
        soltab_names = []
        for soltab_name in iter(list(self.obj._v_groups.keys())):
            soltab_names.append(soltab_name)
        return soltab_names

    def get_soltab(self, soltab, use_cache=False, sel={}):
        """
        Get a soltab with a given name.

        Parameters
        ----------
        soltab : str
            A solution table name.
        use_cache : bool, optional
            Soltabs obj will use cache, by default False.
        sel : dict, optional
            Selection dict, by default no selection.

        Returns
        -------
        soltab obj
            A solution table obj.
        """
        if soltab is None:
            raise Exception(
                "Solution-table not specified while querying for"
                " solution-table."
            )

        if soltab not in self.get_soltab_names():
            raise Exception(
                f"Solution-table {soltab}  not found in solset {self.name}."
            )

        return Soltab(self.obj._f_get_child(soltab), use_cache, sel)

    def get_ant(self):
        """
        Get the antenna subtable with antenna names and positions.

        Returns
        -------
        dict
            Available antennas in the form {name1:[position coords],
            name2:[position coords], ...}.
        """
        ants = {}
        try:
            for ant in self.obj.antenna:
                ants[ant["name"].decode()] = ant["position"]
        except Exception:
            pass

        return ants

    def get_source(self):
        """
        Get the source subtable with direction names and coordinates.

        Returns
        -------
        dict
            Available sources in the form {name1:[ra,dec], name2:[ra,dec],
            ...}.
        """
        sources = {}
        try:
            for source in self.obj.source:
                sources[source["name"].decode()] = source["dir"]
        except Exception:
            pass

        return sources

    def get_ant_dist(self, ant=None):
        """
        Get antenna distance to a specified one.

        Parameters
        ----------
        ant : str
            An antenna name.

        Returns
        -------
        str
            Dict of distances to each antenna. The distance with the antenna
            "ant" is 0.
        """
        if ant is None:
            raise Exception("Missing antenna name.")

        ants = self.get_ant()

        if ant not in list(ants.keys()):  # pylint: disable=C0201
            raise Exception(f"Missing antenna {ant} in antenna table.")

        return {
            a: np.sqrt(
                (loc[0] - ants[ant][0]) ** 2
                + (loc[1] - ants[ant][1]) ** 2
                + (loc[2] - ants[ant][2]) ** 2
            )
            for a, loc in ants.items()
        }


class Soltab:
    """
    Parameters
    ----------
    soltab : pytables Table obj
        Pytable Table object.
    use_cache : bool, optional
        Cache all data in memory, by default False.
    **args : optional
        Used to create a selection.
        Selections examples:
        axis_name = None # to select all
        axis_name = xxx # to select ONLY that value for an axis
        axis_name = [xxx, yyy, zzz] # to selct ONLY those values for an axis
        axis_name = 'xxx' # regular expression selection
        axis_name = {min: xxx} # to selct values grater or equal than xxx
        axis_name = {max: yyy} # to selct values lower or equal than yyy
        axis_name = {min: xxx, max: yyy} # to selct values greater or equal
                   than xxx and lower or equal than yyy
    """

    def __init__(self, soltab, use_cache=False, args={}):

        if not isinstance(soltab, tables.Group):
            logging.error(
                "Object must be initialized with a pyTables Table object."
            )
            sys.exit(1)

        self.obj = soltab
        self.name = soltab._v_name

        # list of axes names, set once to speed up calls
        axes_names_in_h5 = soltab.val.attrs["AXES"].decode()
        self.axes_names = axes_names_in_h5.split(",")

        # dict of axes values,set once to speed up calls (a bit of memory
        # usage)
        self.axes = {}
        for axis in self.get_axes_names():
            self.axes[axis] = soltab._f_get_child(axis)

        # initialize selection
        self.set_selection(**args)

        self.use_cache = use_cache
        if self.use_cache:
            logging.debug("Caching...")
            self.set_cache(self.obj.val, self.obj.weight)

        self.fully_flagged_ants = (
            None  # this is populated if required by reference
        )

    def delete(self):
        """
        Delete this soltab.
        """
        logging.info('Soltab "%s" deleted.', self.name)
        self.obj._f_remove(recursive=True)

    def rename(self, newname, overwrite=False):
        """
        Rename this soltab.

        Parameters
        ----------
        newname : str
            New solution table name.
        overwrite : bool, optional
            Overwrite existing soltab with same name.
        """
        self.obj._f_rename(newname, overwrite)
        logging.info('Soltab "%s" renamed to "%s".', self.name, newname)
        self.name = self.obj._v_name

    def set_cache(self, val, weight):
        """
        Set cache values.

        Parameters
        ----------
        val : array
        weight : array
        """
        self.cache_val = np.copy(val)
        self.cache_weight = np.copy(weight)

    def get_solset(self):
        """
        This is used to obtain the parent solset object to e.g. get antennas
        or create new soltabs.

        Returns
        -------
        solset obj
            A solset obj.
        """
        return Solset(self.obj._v_parent)

    def get_address(self):
        """
        Get the "solset000/soltab000" type string for this Soltab.

        Returns
        -------
        str
            The solset/soltab address of self.obj as a string.
        """
        return self.obj._v_pathname[1:]

    def clear_selection(self):
        """
        Clear selection, all values are now considered.
        """
        self.set_selection()

    def set_selection(self, update=False, **args):
        """
        Set a selection criteria. For each axes there can be a:
            * string: regexp
            * list: use only the listed values
            * dict: with min/max/[step] to select a range.

        Parameters
        ----------
        **args :
            Valid axes names of the form: pol='XX',
            ant=['CS001HBA','CS002HBA'],
            time={'min':1234,'max':'2345','step':4}.

        update : bool
            Only update axes passed as arguments, the rest is maintained.
            Default: False.

        """
        # create an initial selection which selects all values
        if not update:
            self.selection = [slice(None)] * len(self.get_axes_names())

        for axis, sel_val in iter(list(args.items())):
            # if None continue and keep all the values
            if sel_val is None:
                continue
            if axis not in self.get_axes_names():
                logging.warning(
                    "Cannot select on axis %s, it doesn't exist. Ignored.",
                    axis,
                )
                continue

            # find the index of the working axis
            idx = self.get_axes_names().index(axis)

            # slice -> let the slice be as it is
            if isinstance(sel_val, slice):
                self.selection[idx] = sel_val

            # string -> regular expression
            elif isinstance(sel_val, str):
                if not self.get_axis_type(axis).char == "S":
                    logging.warning(
                        'Cannot select on axis "%s" with a regular expression.'
                        " Use all available values.",
                        axis,
                    )
                    continue
                self.selection[idx] = [
                    i
                    for i, item in enumerate(self.get_axis_values(axis))
                    if re.search(sel_val, item)
                ]

                # transform list of 1 element in a relative slice(), faster as
                # it gets reference
                if len(self.selection[idx]) == 1:
                    self.selection[idx] = slice(
                        self.selection[idx][0], self.selection[idx][0] + 1
                    )

            # dict -> min max
            elif isinstance(sel_val, dict):
                axis_vals = self.get_axis_values(axis)
                # some checks
                if "min" in sel_val and sel_val["min"] > np.max(axis_vals):
                    logging.error(
                        "Selection with min > than maximum value. Use all"
                        " available values."
                    )
                    continue
                if "max" in sel_val and sel_val["max"] < np.min(axis_vals):
                    logging.error(
                        "Selection with max < than minimum value. Use all"
                        " available values."
                    )
                    continue

                if "min" in sel_val and "max" in sel_val:
                    self.selection[idx] = slice(
                        np.where(axis_vals >= sel_val["min"])[0][0],
                        np.where(axis_vals <= sel_val["max"])[0][-1] + 1,
                    )
                    # thisSelection[idx] =
                    # list(np.where((axis_vals>=sel_val['min']) &
                    # (axis_vals<=sel_val['max']))[0])

                elif "min" in sel_val:
                    self.selection[idx] = slice(
                        np.where(axis_vals >= sel_val["min"])[0][0], None
                    )
                    # thisSelection[idx] =
                    # list(np.where(axis_vals>=sel_val['min'])[0])
                elif "max" in sel_val:
                    self.selection[idx] = slice(
                        0, np.where(axis_vals <= sel_val["max"])[0][-1] + 1
                    )
                    # thisSelection[idx] =
                    # list(np.where(axis_vals<=sel_val['max'])[0])
                else:
                    logging.error(
                        "Selection with a dict must have 'min' and/or 'max'"
                        " entry. Use all available values."
                    )
                    continue
                if "step" in sel_val:
                    self.selection[idx] = slice(
                        self.selection[idx].start,
                        self.selection[idx].stop,
                        sel_val["step"],
                    )
                    # thisSelection[idx] =
                    # thisSelection[idx][::sel_val['step']]

            # single val/list -> exact matching
            else:
                if isinstance(sel_val, np.ndarray):
                    sel_val = sel_val.tolist()
                if not isinstance(sel_val, list):
                    sel_val = [sel_val]
                # convert to correct data type (from parset everything is a
                # string)
                if not self.get_axis_type(axis).type is np.string_:
                    sel_val = np.array(sel_val, dtype=self.get_axis_type(axis))
                else:
                    sel_val = np.array(sel_val)

                if len(sel_val) == 1:
                    # speedup in the common case of a single value
                    if not sel_val[0] in self.get_axis_values(axis).tolist():
                        logging.error(
                            "Cannot find value %s"  # pylint: disable=C0209
                            " in axis %s. Skip selection.",
                            sel_val[0],
                            axis,
                        )
                        return
                    self.selection[idx] = [
                        self.get_axis_values(axis).tolist().index(sel_val)
                    ]
                else:
                    self.selection[idx] = [
                        i
                        for i, item in enumerate(self.get_axis_values(axis))
                        if item in sel_val
                    ]

                # transform list of 1 element in a relative slice(), faster as
                # it gets a reference
                if len(self.selection[idx]) == 1:
                    self.selection[idx] = slice(
                        self.selection[idx][0], self.selection[idx][0] + 1
                    )
                # transform list of continuous numbers in slices, faster as
                # it gets a reference
                elif (
                    len(self.selection[idx]) != 0
                    and len(self.selection[idx]) - 1
                    == self.selection[idx][-1] - self.selection[idx][0]
                ):
                    self.selection[idx] = slice(
                        self.selection[idx][0], self.selection[idx][-1] + 1
                    )

            # if a selection return an empty list
            # (maybe because of a wrong name), then use all values
            if (
                isinstance(self.selection[idx], list)
                and len(self.selection[idx]) == 0
            ):
                logging.warning(
                    'Empty/wrong selection on axis "%s". Use all available'
                    " values.",
                    axis,
                )
                self.selection[idx] = slice(None)

    def get_type(self):
        """
        Get the solution type of this Soltab.

        Returns
        -------
        str
            Return the type of the solution-tables (e.g. amplitude).
        """

        return self.obj._v_title

    def get_axes_names(self):
        """
        Get axes names.

        Returns
        -------
        list
            A list with all the axis names in the correct order for
            slicing the getValuesGrid() reurned matrix.
        """

        return self.axes_names[:]

    def get_axis_len(self, axis, ignore_selection=False):
        """
        Return an axis lenght.

        Parameters
        ----------
        axis : str
            The name of the axis.
        ignore_selection : bool, optional
            If True returns the axis lenght without any selection active,
            by default False.

        Returns
        -------
        int
            The axis lenght.
        """
        return len(
            self.get_axis_values(axis, ignore_selection=ignore_selection)
        )

    def get_axis_type(self, axis):
        """
        Return the axis dtype

        Parameters
        ----------
        axis : str
            The name of the axis.

        Returns
        -------
        dtype
            The axis dtype.
        """
        if axis not in self.get_axes_names():
            logging.error('Axis "%s" not found.', axis)
            return None

        return self.obj._f_get_child(axis).dtype

    def get_axis_values(self, axis, ignore_selection=False):
        """
        Get the values of a given axis.

        Parameters
        ----------
        axis : str
            The name of the axis.
        ignore_selection : bool, optional
            If True returns the axis values without any selection active,
            by default False.

        Returns
        -------
        list
            A copy of all values present along a specific axis.
        """
        if axis not in self.get_axes_names():
            logging.error('Axis "%s" not found.', axis)
            return None

        if ignore_selection:
            axisvalues = np.copy(self.axes[axis])
        else:
            axis_idx = self.get_axes_names().index(axis)
            axisvalues = np.copy(self.axes[axis][self.selection[axis_idx]])

        if axisvalues.dtype.str[0:2] == "|S":
            # Convert to native string format for python 3
            return axisvalues.astype(str)
        return axisvalues

    def set_axis_values(self, axis, vals):
        """
        Set the value of a specific axis

        Parameters
        ----------
        axis : str
            The name of the axis.
        vals : array
            Values
        """

        if axis not in self.get_axes_names():
            logging.error('Axis "%s" not found.', axis)
            return

        axis_idx = self.get_axes_names().index(axis)
        self.axes[axis][self.selection[axis_idx]] = vals

    def set_values(self, vals, selection=None, weight=False):
        """
        Save values in the val grid

        Parameters
        ----------
        vals : array, float
            values to write as an n-dimentional array which match the
            selection dimention
            if a float is passed or the selected data are set to that value

        selection : selection format, optional
            To set only a subset of data, overriding global selectioan,
            by default use global selection.
            This is used to set values in a loop of getValueIter().
            Global seclection is NOT overwritten.

        weight : bool, optional
            If true store in the weights instead that in the vals,
            by default False
        """
        if selection is None:
            selection = self.selection

        if self.use_cache:
            data_vals = self.cache_weight if weight else self.cache_val
        else:
            data_vals = self.obj.weight if weight else self.obj.val

        # NOTE: pytables has a nasty limitation that only one list can be
        # applied when selecting.
        # Conversely, one can apply how many slices he wants.
        # Single values/contigous values are converted in slices in H5parm.
        # This try/except implements a workaround for this limitation.
        # Once the pytables will be updated, the except can be removed.
        try:
            # the float check allows quick reset of large arrays to a
            # single value
            if isinstance(vals, (np.floating, float)):
                data_vals[tuple(selection)] = vals
            # the reshape is needed when saving e.g. [512] (vals shape)
            # into [512,1,1] (selection output)
            else:
                data_vals[tuple(selection)] = np.reshape(
                    vals, data_vals[tuple(selection)].shape
                )
        except Exception:
            # logging.debug('Optimizing selection writing '+str(selection))
            selection_lists_idx = [
                i for i, s in enumerate(selection) if isinstance(s, list)
            ]
            sub_selection = selection[:]
            # create a sub_selection also for the "vals" array
            sub_selection_for_vals = [
                slice(None) for i in range(len(sub_selection))
            ]
            # cycle across lists and save data index by index
            for selection_list_vals_iter in itertools.product(
                *[
                    selection[selection_list_idx]
                    for selection_list_idx in selection_lists_idx[1:]
                ]
            ):
                for i, selection_list_idx in enumerate(
                    selection_lists_idx[1:]
                ):
                    # this is the sub selection which has a slice for every
                    # slice and a single value for every list
                    sub_selection[
                        selection_list_idx
                    ] = selection_list_vals_iter[i]
                    sub_selection_for_vals[selection_list_idx] = i
                if isinstance(vals, float):
                    data_vals[tuple(sub_selection)] = vals
                else:
                    data_vals[tuple(sub_selection)] = vals[
                        tuple(sub_selection_for_vals)
                    ]

    def flush(self):
        """
        Copy cached values into the table
        """
        if not self.use_cache:
            logging.error("Flushing non cached data.")
            sys.exit(1)

        logging.info("Writing results...")
        self.obj.weight[:] = self.cache_weight
        self.obj.val[:] = self.cache_val

    def __getattr__(self, axis):
        """
        Links any attribute with an "axis name" to getValuesAxis("axis name")
        also links val and weight to the relative arrays.

        Parameters
        ----------
        axis : str
            The axis name.
        """
        if axis == "val":
            return self.get_values(ret_axes_vals=False)
        if axis == "weight":
            return self.get_values(ret_axes_vals=False, weight=True)
        if axis in self.get_axes_names():
            return self.get_axis_values(axis)
        return object.__getattribute__(self, axis)
        # logging.error("Cannot find axis \""+axis+"\".")
        # return None

    def _apply_adv_selection(self, data, selection):
        # NOTE: pytables has a nasty limitation that only one list can be
        # applied when selecting.
        # Conversely, one can apply how many slices he wants.
        # Single values/contigous values are converted in slices in H5parm.
        # This implements a workaround for this limitation. Once the pytables
        # will be updated, the except can be removed.
        if np.sum(
            [1.0 for sel in selection if isinstance(sel, list)]
        ) > 1 and (
            isinstance(data, np.ndarray)
            or np.sum(
                [len(sel) - 1 for sel in selection if isinstance(sel, list)]
            )
            > 0
        ):

            # logging.debug('Optimizing selection reading '+str(selection))
            # for performances is important to minimize the fetched data
            # move all slices at the first selection and lists afterwards
            # (first list is allowd in firstselection)
            selection_lists_idx = [
                i for i, s in enumerate(selection) if isinstance(s, list)
            ]
            first_selection = selection[:]
            for i in selection_lists_idx[1:]:
                first_selection[i] = slice(None)
            # create a second selection using np.ix_
            second_selection = []
            for i, sel in enumerate(selection):
                # if i == selection_lists_idx[0]:
                # second_selection.append(range(
                # self.get_axis_len(self.get_axes_names()[i],
                # ignore_selection=False)))
                if i == selection_lists_idx[0]:
                    second_selection.append(list(range(len(sel))))
                elif isinstance(sel, list):
                    second_selection.append(sel)
                elif isinstance(sel, slice):
                    second_selection.append(
                        list(
                            range(
                                self.get_axis_len(
                                    self.get_axes_names()[i],
                                    ignore_selection=False,
                                )
                            )
                        )
                    )
            # print first_selection
            # print second_selection
            # print data[tuple(first_selection)].shape
            # print
            # data[tuple(first_selection)][np.ix_(*second_selection)].shape
            return data[tuple(first_selection)][np.ix_(*second_selection)]
        return data[tuple(selection)]

    def _get_fully_flagged_ants(self):
        if self.fully_flagged_ants is None:
            self.fully_flagged_ants = []  # fully flagged antennas
            ant_axis = self.get_axes_names().index("ant")

            if self.use_cache:
                data_weights = self.cache_weight
            else:
                data_weights = self.obj.weight

            for ant_to_check in self.get_axis_values(
                "ant", ignore_selection=True
            ):
                # fully flagged?
                ref_selection = [slice(None)] * len(self.get_axes_names())
                ref_selection[ant_axis] = [
                    self.get_axis_values("ant", ignore_selection=True)
                    .tolist()
                    .index(ant_to_check)
                ]
                if (
                    self._apply_adv_selection(data_weights, ref_selection) == 0
                ).all():
                    self.fully_flagged_ants.append(ant_to_check)

        return self.fully_flagged_ants

    @deprecated_alias(
        reference="ref_ant"
    )  # Add alias for backwards compatibility
    def get_values(
        self, ret_axes_vals=True, weight=False, ref_ant=None, ref_dir=None
    ):
        """
        Creates a simple matrix of values. Fetching a copy of all selected rows
        into memory.

        Parameters
        ----------
        ret_axes_vals : bool, optional
            If true returns also the axes vals as a dict of:
            {'axisname1':[axisvals1],'axisname2':[axisvals2],...}.
            By default True.
        weight : bool, optional
            If true get the weights instead that the vals, by defaul False.
        ref_ant : str, optional
            In case of phase or rotation solutions, reference to this station
            name.
            By default no reference.
            If "closest" reference to the closest antenna.
        ref_dir : str, optional
            In case of phase or rotation solutions, reference to this Direction
            By default no reference.
            If "center", reference to the central direction.

        Returns
        -------
        array
            A numpy ndarray (values or weights depending on parameters)
            If selected, returns also the axes values
        """
        if self.use_cache:
            if weight:
                data_vals = self.cache_weight
            else:
                data_vals = self.cache_val
        else:
            if weight:
                data_vals = self.obj.weight
            else:
                data_vals = self.obj.val

        data_vals = self._apply_adv_selection(data_vals, self.selection)

        # CASE 1: Reference only to ant
        if ref_ant and not ref_dir:
            # TODO: Should there be a warning if only ant is referenced but
            # multiple directions are present?
            if not self.get_type() in [
                "phase",
                "scalarphase",
                "rotation",
                "tec",
                "clock",
                "tec3rd",
                "rotationmeasure",
            ]:
                logging.error(
                    "Reference possible only for phase, scalarphase, clock,"
                    " tec, tec3rd, rotation and rotationmeasure solution"
                    " tables. Ignore referencing."
                )
            elif "ant" not in self.get_axes_names():
                logging.error(
                    "Cannot find antenna axis for referencing phases. Ignore"
                    " referencing."
                )
            elif (
                ref_ant
                not in self.get_axis_values("ant", ignore_selection=True)
                and ref_ant != "closest"
            ):
                logging.error(
                    "Cannot find antenna %s. Ignore referencing.", ref_ant
                )
            else:
                if self.use_cache:
                    if weight:
                        data_vals_ref = self.cache_weight
                    else:
                        data_vals_ref = self.cache_val
                else:
                    if weight:
                        data_vals_ref = self.obj.weight
                    else:
                        data_vals_ref = self.obj.val

                ant_axis = self.get_axes_names().index("ant")
                ref_selection = self.selection[:]

                if ref_ant == "closest":
                    # put antenna axis first
                    data_vals = np.swapaxes(data_vals, 0, ant_axis)

                    for i, ant_to_ref in enumerate(
                        self.get_axis_values("ant")
                    ):
                        # get the closest antenna
                        ant_dists = self.get_solset().get_ant_dist(
                            ant_to_ref
                        )  # this is a dict
                        for bad_ant in self._get_fully_flagged_ants():
                            del ant_dists[bad_ant]  # remove bad ants

                        ref_ant = list(ant_dists.keys())[
                            list(ant_dists.values()).index(
                                sorted(ant_dists.values())[1]
                            )
                        ]
                        # get the second closest antenna (the first is itself)

                        ref_selection[ant_axis] = [
                            self.get_axis_values("ant", ignore_selection=True)
                            .tolist()
                            .index(ref_ant)
                        ]
                        data_vals_ref_i = self._apply_adv_selection(
                            data_vals_ref, ref_selection
                        )
                        data_vals_ref_i = np.swapaxes(
                            data_vals_ref_i, 0, ant_axis
                        )
                        if weight:
                            data_vals[i][data_vals_ref_i[0] == 0.0] = 0.0
                        else:
                            data_vals[i] -= data_vals_ref_i[0]

                    data_vals = np.swapaxes(data_vals, 0, ant_axis)

                else:
                    ref_selection[ant_axis] = [
                        self.get_axis_values("ant", ignore_selection=True)
                        .tolist()
                        .index(ref_ant)
                    ]
                    data_vals_ref = self._apply_adv_selection(
                        data_vals_ref, ref_selection
                    )

                    if weight:
                        data_vals[
                            np.repeat(
                                data_vals_ref,
                                axis=ant_axis,
                                repeats=len(self.get_axis_values("ant")),
                            )
                            == 0.0
                        ] = 0.0
                    else:
                        data_vals = data_vals - np.repeat(
                            data_vals_ref,
                            axis=ant_axis,
                            repeats=len(self.get_axis_values("ant")),
                        )
                # if not weight and not self.get_type() != 'tec' and not
                # self.get_type() != 'clock' and not self.get_type() !=
                # 'tec3rd' and not self.get_type() != 'rotationmeasure':
                #     data_vals = normalize_phase(data_vals)
        # CASE 2: Reference only to dir
        # TODO: should there be a warning if only direction is referenced but
        # multipled ants are present?
        elif ref_dir and not ref_ant:
            if not self.get_type() in [
                "phase",
                "scalarphase",
                "rotation",
                "tec",
                "clock",
                "tec3rd",
                "rotationmeasure",
            ]:
                logging.error(
                    "Reference possible only for phase, scalarphase, clock,"
                    " tec, tec3rd, rotation and rotationmeasure solution"
                    " tables. Ignore referencing."
                )
            elif "dir" not in self.get_axes_names():
                logging.error(
                    "Cannot find direction axis for referencing phases. Ignore"
                    " referencing."
                )
            elif (
                ref_dir
                not in self.get_axis_values("dir", ignore_selection=True)
                and ref_dir != "center"
            ):
                logging.error(
                    "Cannot find direction %s. Ignore referencing.", ref_dir
                )
            else:
                if self.use_cache:
                    if weight:
                        data_vals_ref = self.cache_weight
                    else:
                        data_vals_ref = self.cache_val
                else:
                    if weight:
                        data_vals_ref = self.obj.weight
                    else:
                        data_vals_ref = self.obj.val

                dir_axis = self.get_axes_names().index("dir")
                ref_selection = self.selection[:]

                if ref_dir == "center":
                    # get the center (=closest to average) direction
                    dirs_dict = self.get_solset().get_source()
                    mean_dir = np.mean([dirs_dict.items()], axis=0)
                    ref_dir, _ = min(
                        dirs_dict.items(),
                        key=lambda kd: np.linalg.norm(kd[1] - mean_dir),
                    )

                ref_selection[dir_axis] = [
                    self.get_axis_values("dir", ignore_selection=True)
                    .tolist()
                    .index(ref_dir)
                ]
                data_vals_ref = self._apply_adv_selection(
                    data_vals_ref, ref_selection
                )

                if weight:
                    data_vals[
                        np.repeat(
                            data_vals_ref,
                            axis=dir_axis,
                            repeats=len(self.get_axis_values("dir")),
                        )
                        == 0.0
                    ] = 0.0
                else:
                    data_vals = data_vals - np.repeat(
                        data_vals_ref,
                        axis=dir_axis,
                        repeats=len(self.get_axis_values("dir")),
                    )
                # if not weight and not self.get_type() != 'tec'
                # and not self.get_type() != 'clock' and not self.get_type() !=
                # 'tec3rd' and not self.get_type() != 'rotationmeasure':
                #     data_vals = normalize_phase(data_vals)

        # CASE 3: Reference to ant and to dir
        if ref_ant and ref_dir:
            if not self.get_type() in [
                "phase",
                "scalarphase",
                "rotation",
                "tec",
                "clock",
                "tec3rd",
                "rotationmeasure",
            ]:
                logging.error(
                    "Reference possible only for phase, scalarphase, clock,"
                    " tec, tec3rd, rotation and rotationmeasure solution"
                    " tables. Ignore referencing."
                )
            elif "ant" not in self.get_axes_names():
                logging.error(
                    "Cannot find antenna axis for referencing phases. Ignore"
                    " referencing."
                )
            elif "dir" not in self.get_axes_names():
                logging.error(
                    "Cannot find direction axis for referencing phases. Ignore"
                    " referencing."
                )
            elif (
                ref_ant
                not in self.get_axis_values("ant", ignore_selection=True)
                and ref_ant != "closest"
            ):
                logging.error(
                    "Cannot find antenna %s. Ignore referencing.", ref_ant
                )
            elif ref_ant == "closest":  # TODO: This needs to be implemented...
                logging.error(
                    "ref_ant='closest' is not supported (yet) when also"
                    " referencing a direction."
                )
            elif (
                ref_dir
                not in self.get_axis_values("dir", ignore_selection=True)
                and ref_dir != "center"
            ):
                logging.error(
                    "Cannot find direction %s. Ignore referencing.", ref_dir
                )
            else:
                if self.use_cache:
                    if weight:
                        data_vals_ref = self.cache_weight
                    else:
                        data_vals_ref = self.cache_val
                else:
                    if weight:
                        data_vals_ref = self.obj.weight
                    else:
                        data_vals_ref = self.obj.val

                ant_axis = self.get_axes_names().index("ant")
                dir_axis = self.get_axes_names().index("dir")
                ref_selection = self.selection[:]

                if ref_dir == "center":
                    # get the center (=closest to average) direction
                    dirs_dict = self.get_solset().get_source()
                    mean_dir = np.mean([dirs_dict.items()], axis=0)
                    ref_dir, _ = min(
                        dirs_dict.items(),
                        key=lambda kd: np.linalg.norm(kd[1] - mean_dir),
                    )

                ref_selection[ant_axis] = [
                    self.get_axis_values("ant", ignore_selection=True)
                    .tolist()
                    .index(ref_ant)
                ]
                ref_selection[dir_axis] = [
                    self.get_axis_values("dir", ignore_selection=True)
                    .tolist()
                    .index(ref_dir)
                ]

                data_vals_ref = self._apply_adv_selection(
                    data_vals_ref, ref_selection
                )

                if weight:
                    data_vals_ref = np.repeat(
                        data_vals_ref, data_vals.shape[ant_axis], axis=ant_axis
                    )
                    data_vals_ref = np.repeat(
                        data_vals_ref, data_vals.shape[dir_axis], axis=dir_axis
                    )
                    data_vals[data_vals_ref == 0.0] = 0.0

                else:
                    data_vals = data_vals - data_vals_ref
                    # np.expand_dims(data_vals_ref, axis=(ant_axis,dir_axis))

        if not ret_axes_vals:
            return data_vals

        axis_vals = {}
        for axis in self.get_axes_names():
            axis_vals[axis] = self.get_axis_values(axis)

        return data_vals, axis_vals

    @deprecated_alias(
        reference="ref_ant"
    )  # Add alias for backwards compatibility
    def get_values_iter(
        self, return_axes=[], weight=False, ref_ant=None, ref_dir=None
    ):
        """
        Return an iterator which yields the values matrix
        (with axes = return_axes) iterating along the other axes.
        E.g. if return_axes are ['freq','time'], one gets a interetion over
        all the possible NxM matrix where N are the freq and M the time
        dimensions. The other axes are iterated in the get_axes_names() order.
        Note that all the data are fetched in memory before returning
        them one at a time. This is quicker.

        Parameters
        ----------
        return_axes : list
            Axes of the returned array, all _others_ will be cycled on each
            element combinations.
        weight : bool, optional
            If true return also the weights, by default False.
        ref_ant : str
            In case of phase solutions, reference to this station name.
        ref_dir : str
            In case of phase solutions, reference to this direction name.

        Returns
        -------
        1) data ndarray of dim=dim(return_axes) and with the axes ordered as
        in get_axes_names()
        2) (if weight == True) weigth ndarray of dim=dim(return_axes) and with
        the axes ordered as in get_axes_names()
        3) a dict with axis values in the form:
        {'axisname1':[axisvals1],'axisname2':[axisvals2],...}
        4) a selection which should be used to write this data back using a
        set_values()
        """
        if weight:
            weigth_vals = self.get_values(
                ret_axes_vals=False,
                weight=True,
                ref_ant=ref_ant,
                ref_dir=ref_dir,
            )
        data_vals = self.get_values(
            ret_axes_vals=False, weight=False, ref_ant=ref_ant, ref_dir=ref_dir
        )

        # get dimensions of non-returned axis (in correct order)
        iter_axes_dim = [
            self.get_axis_len(axis)
            for axis in self.get_axes_names()
            if axis not in return_axes
        ]

        # generator to cycle over all the combinations of iterAxes
        # it "simply" gets the indexes of this particular combination of
        # iterAxes and use them to refine the selection.
        def generator():
            for axis_idx in np.ndindex(tuple(iter_axes_dim)):
                ref_selection = []
                return_selection = []
                this_axes_vals = {}
                i = 0
                for j, axis_name in enumerate(self.get_axes_names()):
                    if axis_name in return_axes:
                        this_axes_vals[axis_name] = self.get_axis_values(
                            axis_name
                        )
                        # add a slice with all possible values (main selection
                        # is preapplied)
                        ref_selection.append(slice(None))
                        # for the return selection use the "main" selection for
                        #  the return axes
                        return_selection.append(self.selection[j])
                    else:
                        # TODO: the iteration axes are not into a 1 element
                        # array, is it a problem?
                        this_axes_vals[axis_name] = self.get_axis_values(
                            axis_name
                        )[axis_idx[i]]
                        # add this index to the refined selection, this will
                        # return a single value for this axis
                        # an int is appended, this will remove an axis from the
                        #  final data
                        ref_selection.append(axis_idx[i])
                        # for the return selection use the complete axis and
                        # find the correct index
                        return_selection.append(
                            [
                                self.get_axis_values(
                                    axis_name, ignore_selection=True
                                )
                                .tolist()
                                .index(this_axes_vals[axis_name])
                            ]
                        )
                        i += 1

                # costly command
                data = data_vals[tuple(ref_selection)]
                if weight:
                    weights = weigth_vals[tuple(ref_selection)]
                    yield (data, weights, this_axes_vals, return_selection)
                else:
                    yield (data, this_axes_vals, return_selection)

        return generator()

    def add_history(self, entry, date=True):
        """
        Adds entry to the table history with current date and time

        Since attributes cannot by default be larger than 64 kB, each
        history entry is stored in a separate attribute.

        Parameters
        ----------
        entry : str
            entry to add to history list
        """

        current_time = str(datetime.datetime.now()).split(".", maxsplit=1)[0]
        attrs = self.obj.val.attrs._f_list("user")
        nums = []
        for attr in attrs:
            try:
                if attr[:-3] == "HISTORY":
                    nums.append(int(attr[-3:]))
            except Exception:
                pass
        history_attr_str = min(list(set(range(1000)) - set(nums)))
        history_attr = f"HISTORY{history_attr_str:03d}"

        if date:
            entry = current_time + ": " + str(entry)
        else:
            entry = str(entry)

        self.obj.val.attrs[history_attr] = entry.encode()

    def get_history(self):
        """
        Get the soltab history.

        Returns
        -------
        str
            The table history as a string with each entry separated by newlines
        """
        attrs = self.obj.val.attrs._f_list("user")
        attrs.sort()
        history_list = []
        for attr in attrs:
            if attr[:-3] == "HISTORY":
                history_list.append(self.obj.val.attrs[attr].decode())

        return "" if len(history_list) == 0 else "\n".join(history_list)
