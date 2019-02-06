import sqlite3
conn = sqlite3.connect('images.db')

c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS image
                 (url TEXT, protein TEXT, antibody TEXT, cell_line TEXT, location TEXT)''')

# Prevent Duplicate Entries
c.execute('''CREATE UNIQUE INDEX IF NOT EXISTS image_no_dups ON image(url, protein, antibody, cell_line, location)''')

# Allow Quick Search By Protein
c.execute('''CREATE INDEX IF NOT EXISTS image_protein ON image (protein)''')

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
