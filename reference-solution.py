"""Reference entry point for the safe SDK application.

The implementation lives in ``robot_app.py`` so the workshop starter and
reference commands cannot drift into two different robot-control paths.
"""

from robot_app import main

if __name__ == "__main__":
    main()
