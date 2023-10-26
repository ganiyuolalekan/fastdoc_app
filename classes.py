import json
import sqlite3

from pydantic import BaseModel, Field
from variables import title_desc, generated_text_desc


class GenerationModel(BaseModel):
    """Speaker names in the given transcript"""

    title: str = Field(..., description=title_desc)
    generated_text: str = Field(..., description=generated_text_desc)


class ClassDatabase:
    """A database for storing class objects."""

    def __init__(self, db_file):
        self.db_file = db_file
        self._connection = sqlite3.connect(db_file)

        try:
            self._create_table()
        except sqlite3.OperationalError:
            pass

    def add_class(self, class_object, _id):
        """Adds a class object to the database."""

        try:
            self._connection.execute(
                "INSERT INTO classes (id, class) VALUES (?, ?)", (_id, json.dumps(class_object)))
        except sqlite3.IntegrityError as e:
            self.update_class(_id, class_object)

    def get_class(self, _id):
        """Returns the class object with the given id."""

        cursor = self._connection.execute("SELECT class FROM classes WHERE id = ?", (_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def update_class(self, _id, class_object):
        """Updates the class object with the given id."""

        self._connection.execute(
            "UPDATE classes SET class = ? WHERE id = ?", (json.dumps(class_object), _id))

    def delete_class(self, _id):
        """Deletes the project with the given id."""

        self._connection.execute("DELETE FROM classes WHERE id = ?", (_id,))

    def _create_table(self):
        """Creates the classes table in the database."""

        self._connection.execute("""CREATE TABLE classes (
          id INTEGER PRIMARY KEY,
          class TEXT
        )""")

    def close(self):
        """Closes the database connection."""

        self._connection.close()
