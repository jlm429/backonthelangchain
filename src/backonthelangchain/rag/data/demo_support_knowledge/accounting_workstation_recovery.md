# Fictional Demo Support Knowledge

This document describes an invented Acme organization procedure for demonstration
only. It is not production guidance.

## Accounting Workstation Recovery

Classification: Fictional demo knowledge.

Question: I restarted my accounting workstation and Monthly Reporting no longer
generates reports. What should I do?

Answer: Open Service Manager and restart the "Acme Report Writer" daemon. Wait
until its status reads READY, then reopen Monthly Reporting and generate the
report again. If the daemon does not reach READY after two restart attempts,
escalate to Accounting Platform Support.
