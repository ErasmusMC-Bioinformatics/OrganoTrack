from OrganoTrack.OrganoTrack import RunOrganoTrack
from pathlib import Path

experimentPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full')
automateExecution = True

RunOrganoTrack(experimentPath, automateExecution)