#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import textwrap
import uuid
from collections import OrderedDict
import warnings

from sacred.__about__ import __version__
from sacred.commandline_options import cli_option
from sacred.observers import RunObserver

import itertools

flatten = lambda i: list(itertools.chain.from_iterable(i))

class SuperTinyDbObserver(RunObserver):

    VERSION = "TinyDbObserver-{}".format(__version__)

    @classmethod
    def create(cls, path="./runs_db", overwrite=None):
        warnings.warn(
            "TinyDbObserver.create(...) is deprecated. "
            "Please use TinyDbObserver(...) instead.",
            DeprecationWarning,
        )
        return cls(path, overwrite)

    def __init__(self, path="./runs_db", overwrite=None):
        from .bases import get_db_file_manager

        root_dir = os.path.abspath(path)
        os.makedirs(root_dir, exist_ok=True)

        db, fs = get_db_file_manager(root_dir, get_all_databases=False)
        self.db = db
        self.runs = db.table("runs")
        self.fs = fs
        self.overwrite = overwrite
        self.run_entry = {}
        self.db_run_id = None
        self.root = root_dir

        print(f"[TinyDBObserver] Currently contains {len(self.runs.all())} entries")

    @classmethod
    def create_from(cls, db, fs, overwrite=None, root=None):
        """Instantiate a TinyDbObserver with an existing db and filesystem."""
        self = cls.__new__(cls)  # skip __init__ call
        self.db = db
        self.runs = db.table("runs")
        self.fs = fs
        self.overwrite = overwrite
        self.run_entry = {}
        self.db_run_id = None
        self.root = root
        return self

    def save(self):
        """Insert or update the current run entry."""
        if self.db_run_id:
            self.runs.update(self.run_entry, doc_ids=[self.db_run_id])
        else:
            db_run_id = self.runs.insert(self.run_entry)
            self.db_run_id = db_run_id

    def save_sources(self, ex_info):

        source_info = []
        for source_name, md5 in ex_info["sources"]:

            # Substitute any HOME or Environment Vars to get absolute path
            abs_path = os.path.join(ex_info["base_dir"], source_name)
            abs_path = os.path.expanduser(abs_path)
            abs_path = os.path.expandvars(abs_path)
            # handle = BufferedReaderWrapper(open(abs_path, "rb"))
            # handle = ArchivableFilepath(abs_path)


            file = self.fs.get(md5)
            if file:
                id_ = file.id
            else:
                address = self.fs.put(abs_path)
                id_ = address.id
            source_info.append([source_name, id_])
        return source_info

    def queued_event(
        self, ex_info, command, host_info, queue_time, config, meta_info, _id
    ):
        raise NotImplementedError(
            "queued_event method is not implemented for" " local TinyDbObserver."
        )

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        self.db_run_id = None

        self.run_entry = {
            "experiment": dict(ex_info),
            "format": self.VERSION,
            "command": command,
            "host": dict(host_info),
            "start_time": start_time,
            "config": config,
            "meta": meta_info,
            "status": "RUNNING",
            "resources": [],
            "artifacts": [],
            "captured_out": "",
            "info": {},
            "heartbeat": None,
        }

        # set ID if not given
        if _id is None:
            _id = uuid.uuid4().hex

        self.run_entry["_id"] = _id

        # save sources
        self.run_entry["experiment"]["sources"] = self.save_sources(ex_info)
        self.save()
        return self.run_entry["_id"]

    def heartbeat_event(self, info, captured_out, beat_time, result):
        self.run_entry["info"] = info
        self.run_entry["captured_out"] = captured_out
        self.run_entry["heartbeat"] = beat_time
        self.run_entry["result"] = result
        self.save()

    def completed_event(self, stop_time, result):
        self.run_entry["stop_time"] = stop_time
        self.run_entry["result"] = result
        self.run_entry["status"] = "COMPLETED"
        self.save()

    def interrupted_event(self, interrupt_time, status):
        self.run_entry["stop_time"] = interrupt_time
        self.run_entry["status"] = status
        self.save()

    def failed_event(self, fail_time, fail_trace):
        self.run_entry["stop_time"] = fail_time
        self.run_entry["status"] = "FAILED"
        self.run_entry["fail_trace"] = fail_trace
        self.save()

    def resource_event(self, filename):
        from .bases import BufferedReaderWrapper

        id_ = self.fs.put(filename).id
        handle = BufferedReaderWrapper(open(filename, "rb"))
        resource = [filename, id_, handle]

        if resource not in self.run_entry["resources"]:
            self.run_entry["resources"].append(resource)
            self.save()

    def artifact_event(self, name, filename, metadata=None, content_type=None):
        from .bases import BufferedReaderWrapper

        id_ = self.fs.put(filename).id
        handle = BufferedReaderWrapper(open(filename, "rb"))
        artifact = [name, filename, id_, handle]

        if artifact not in self.run_entry["artifacts"]:
            self.run_entry["artifacts"].append(artifact)
            self.save()

    def __eq__(self, other):
        if isinstance(other, SuperTinyDbObserver):
            return self.runs.all() == other.runs.all()
        return False


@cli_option("-t", "--tiny_db")
def tiny_db_option(args, run):
    """Add a TinyDB Observer to the experiment.

    The argument is the path to be given to the TinyDbObserver.
    """
    tinydb_obs = SuperTinyDbObserver(path=args)
    run.observers.append(tinydb_obs)


class SuperTinyDbReader:
    def __init__(self, path):
        from .bases import get_db_file_manager

        root_dir = os.path.abspath(path)
        if not os.path.exists(root_dir):
            raise IOError("Path does not exist: %s" % path)

        dbs, fs = get_db_file_manager(root_dir, get_all_databases=True)

        self.dbs = dbs
        self.runs = {name: db.table("runs") for name, db in dbs.items()}
        self.fs = fs

    def search(self, *args, servers_only=False, **kwargs):
        """Wrapper to TinyDB's search function."""
        return flatten([run.search(*args, **kwargs) for run in self._get_runs(servers_only)])

    def fetch_files(self, exp_name=None, query=None, indices=None, servers_only=False):
        """Return Dictionary of files for experiment name or query.

        Returns a list of one dictionary per matched experiment. The
        dictionary is of the following structure

            {
              'exp_name': 'scascasc',
              'exp_id': 'dqwdqdqwf',
              'date': datatime_object,
              'sources': [ {'filename': filehandle}, ..., ],
              'resources': [ {'filename': filehandle}, ..., ],
              'artifacts': [ {'filename': filehandle}, ..., ]
            }

        """
        entries = self.fetch_metadata(exp_name, query, indices, servers_only)

        all_matched_entries = []
        for ent in entries:

            rec = dict(
                exp_name=ent["experiment"]["name"],
                exp_id=ent["_id"],
                date=ent["start_time"],
            )

            source_files = {x[0]: x[2] for x in ent["experiment"]["sources"]}
            resource_files = {x[0]: x[2] for x in ent["resources"]}
            artifact_files = {x[0]: x[3] for x in ent["artifacts"]}

            if source_files:
                rec["sources"] = source_files
            if resource_files:
                rec["resources"] = resource_files
            if artifact_files:
                rec["artifacts"] = artifact_files

            all_matched_entries.append(rec)

        return all_matched_entries

    def fetch_report(self, exp_name=None, query=None, indices=None, servers_only=False):

        template = """
-------------------------------------------------
Experiment: {exp_name}
-------------------------------------------------
ID: {exp_id}
Date: {start_date}    Duration: {duration}
Host: {hostname}

Parameters:
{parameters}

Result:
{result}

Dependencies:
{dependencies}

Resources:
{resources}

Source Files:
{sources}

Outputs:
{artifacts}
"""

        entries = self.fetch_metadata(exp_name, query, indices, servers_only)

        all_matched_entries = []
        for ent in entries:

            date = ent["start_time"]
            weekdays = "Mon Tue Wed Thu Fri Sat Sun".split()
            w = weekdays[date.weekday()]
            date = " ".join([w, date.strftime("%d %b %Y - %H:%M:%S")])

            duration = ent["stop_time"] - ent["start_time"]
            secs = duration.total_seconds()
            hours, remainder = divmod(secs, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration = "%02d:%02d:%04.1f" % (hours, minutes, seconds)

            parameters = self._dict_to_indented_list(ent["config"])

            result = self._indent(ent["result"].__repr__(), prefix="    ")

            deps = ent["experiment"]["dependencies"]
            deps = self._indent("\n".join(deps), prefix="    ")

            resources = [x[0] for x in ent["resources"]]
            resources = self._indent("\n".join(resources), prefix="    ")

            sources = [x[0] for x in ent["experiment"]["sources"]]
            sources = self._indent("\n".join(sources), prefix="    ")

            artifacts = [x[0] for x in ent["artifacts"]]
            artifacts = self._indent("\n".join(artifacts), prefix="    ")

            hostname = ent["host"]["hostname"]

            none_str = "    None"

            rec = dict(
                exp_name=ent["experiment"]["name"],
                exp_id=ent["_id"],
                start_date=date,
                duration=duration,
                parameters=parameters if parameters else none_str,
                result=result if result else none_str,
                dependencies=deps if deps else none_str,
                resources=resources if resources else none_str,
                sources=sources if sources else none_str,
                artifacts=artifacts if artifacts else none_str,
                hostname=hostname if hostname else none_str
            )

            report = template.format(**rec)

            all_matched_entries.append(report)

        return all_matched_entries

    def fetch_metadata(self, exp_name=None, query=None, indices=None, servers_only=False):
        """Return all metadata for matching experiment name, index or query."""
        from tinydb import Query

        if exp_name or query:
            if query:
                q = query
            elif exp_name:
                q = Query().experiment.name.search(exp_name)

            entries = self.search(q, servers_only=servers_only)

        elif indices or indices == 0:
            if not isinstance(indices, (tuple, list)):
                indices = [indices]

            num_recs = len(self._all_run_entries(servers_only))

            for idx in indices:
                if idx >= num_recs:
                    raise ValueError(
                        "Index value ({}) must be less than "
                        "number of records ({})".format(idx, num_recs)
                    )

            entries = [self._all_run_entries(servers_only)[ind] for ind in indices]

        else:
            return self._all_run_entries(servers_only)

            # raise ValueError(
            #     "Must specify an experiment name, indicies or " "pass custom query"
            # )

        return entries

    def _get_runs(self, servers_only):
        # print(f"Getting runs, {self.runs=}")
        # print(f"{self.dbs=}")
        non_servers = ["az", "elp"]  # TODO: or do opposite?
        # print(f"{non_servers=}, {self.runs.keys()=}")

        if not servers_only:
            non_servers.clear()

        # print({name: len(table.all()) for name, table in self.runs.items()})
        return [run for name, run in self.runs.items() if name not in non_servers]


    def _all_run_entries(self, servers_only):
        return sorted(flatten([run.all() for run in self._get_runs(servers_only)]), key=lambda e: e["start_time"])

    def _dict_to_indented_list(self, d):

        d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        output_str = ""

        for k, v in d.items():
            output_str += "%s: %s" % (k, v)
            output_str += "\n"

        output_str = self._indent(output_str.strip(), prefix="    ")

        return output_str

    def _indent(self, message, prefix):
        """Wrapper for indenting strings in Python 2 and 3."""
        preferred_width = 150
        wrapper = textwrap.TextWrapper(
            initial_indent=prefix, width=preferred_width, subsequent_indent=prefix
        )

        lines = message.splitlines()
        formatted_lines = [wrapper.fill(lin) for lin in lines]
        formatted_text = "\n".join(formatted_lines)

        return formatted_text
