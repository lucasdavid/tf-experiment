from sacred.run import Run


def get_run_params(_run):
  report_dir = None

  if _run.observers:
    for o in _run.observers:
      if hasattr(o, 'dir'):
        report_dir = o.dir
        break

  return {'report_dir': report_dir}


__all__ = [
  'Run',
  'get_run_params',
]
