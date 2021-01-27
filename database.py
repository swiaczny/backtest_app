# import os
import pandas as pd
import psycopg2
from psycopg2 import Error

class Connect():
	'''
	postgres database functionalities
	'''
	
	def __init__(self, name):
		database = {
			'homedev': {
				'USER': os.environ.get('DB_USER'),
				'PASSWORD':os.environ.get('DB_PASSWORD'),
				'HOST':os.environ.get('DB_HOST'),
				'PORT':'5432',
			}
		}

		self.dbname = name
		self.user = database[name]['USER']
		self.pw = database[name]['PASSWORD']
		self.host = database[name]['HOST']
		self.port = database[name]['PORT']
		self.connection = psycopg2.connect(
			user = self.user,password = self.pw, 
			host = self.host, 
			port = self.port, 
			database = self.dbname)		
	
	def close(self):

		self.connection.close()

	def _get_table_columns(self, table):
		'''get column names as list from db-table'''
		db_table = self.import_from_db(
			table = 'information_schema.columns', 
			columns = 'column_name',
			condition = f"where table_name ='{table}'"
			)

		return [c for c in db_table['column_name']]
	
	def _get_table_keys(self, table):
		''' extract key-columns from db-table as a list '''
		db_table = self.import_from_db(
			table = 'information_schema.key_column_usage',
			columns = 'column_name',
			condition = f"where table_name = '{table}'" 
			)
		return [c for c in db_table['column_name']]

	def _df_to_tuple(self, df):

		return df.where((pd.notnull(df)), None).to_records(index=False).tolist()

	def _insert(self, table, df):
		'''exports data from df-table'''

		db_columns = self._get_table_columns(table)
		df_columns = df.columns.to_list()

		# only insert data if col-names match
		if(df_columns == db_columns):
						
			sql_cols = ','.join(df_columns)
			sql_signs = ','.join(['%s' for i in range(len(df_columns))])
			
			sql = f'insert into {table} ({sql_cols}) values ({sql_signs})'
			data = self._df_to_tuple(df)

			cur = self.connection.cursor()
			
			if len(data)==1:
				cur.execute(sql, data[0])

			elif len(data)>1:
				cur.executemany(sql, data)

			print('SUCCESS:',cur.rowcount, f'records inserted into table "{table}"')

		else:
			print(f'FAIL: columns in df do not match columns in "{self.dbname}.{table}"')
		
		self.connection.commit()
		cur.close()

	def _insert_new(self, df, table):
		
		data = self._df_to_tuple(df)
		df_columns = df.columns.to_list()
		
		db_columns = self._get_table_columns(table)
		db_keys = self._get_table_keys(table)

		# only insert data if col-names match
		if(df_columns == db_columns):						

			sql_keys = ','.join(db_keys)
			sql_cols = ','.join(df_columns)
			sql_signs = ','.join(['%s' for i in range(len(df_columns))])

			# insert statement
			sql = f'insert into {table} ({sql_cols}) values ({sql_signs})'
			sql += f' on conflict ({sql_keys}) do nothing;'

			cur = self.connection.cursor()
			cur.executemany(sql, data)
			print('SUCCESS:',cur.rowcount, f'records inserted into table "{table}"')
		
		else:
			return print(f'FAIL: columns in df do not match columns in "{self.dbname}.{table}"')

		self.connection.commit()
		cur.close()

	def _delete(self, table, condition=None):
		'''deletes all observations in db-table if condition = None'''

		cur = self.connection.cursor()

		if condition == None:
			# deleting all
			sql = f'truncate table {table}'
		else:
			sql = f'delete from {table} {condition}'

		cur.execute(sql)
		self.connection.commit()
		
		print('SUCCESS:',cur.rowcount, f'records have been deleted from table {table}')
		
		cur.close()

	def init_table(self, table_name, column_names_formats, primary_key):
		# TODO :
		# check if table exists
		# SELECT to_regclass('schema_name.table_name');
		
		# check if inputs are lists or tuples
		if not isinstance(column_names_formats, (list, tuple)):
			cols = ','.join([column_names_formats])
		else:
			cols = ','.join(column_names_formats)

		if not isinstance(primary_key, (list, tuple)):
			keys = ','.join([primary_key])
		else:
			keys = ','.join(primary_key)

		try:
			cur = self.connection.cursor()

			sql = f'create table if not exists {table_name} ({cols});'
			sql += f'alter table {table_name} add primary key ({keys});'

			cur.execute(sql)

			# commit and close 
			self.connection.commit()
			self.connection.cursor().close()

		except (Exception, psycopg2.Error) as error:
			if(self.connection):
				print("ERROR:", error)

	def import_from_db(self, table, columns='*', condition=None, date_cols=None):
		'''
			returns pandas df
		'''
		if columns == '*':
			# extract column names from db
			col_list = self._get_table_columns(table)

		else:
			# one or more columns selected
			if ',' in columns:
				col_list = columns.split(',')
			else:
				col_list = [columns]

		# construct sql
		sql = f'select {columns} from {table}'
		if condition:
			sql += f' {condition}'

		try:
			cur = self.connection.cursor()
			# read
			cur.execute(sql)
			records = cur.fetchall() 

		except (Exception, psycopg2.Error) as error :
			if(self.connection):
				print("ERROR:", error)

		finally:
			cur.close()
			pass
		
		odf = pd.DataFrame(data=records, columns=col_list)

		# change date format to datetime if needed
		if date_cols is not None:
			for d in date_cols:
				odf[d]=pd.to_datetime(odf[d])

		return odf

	def export_to_database(self, df, table, mode):
		''' exports df to db '''

		if mode == 'empty_insert':

			self._delete(table=table)
			self._insert(table=table, df=df)

		elif mode == 'insert_new':
			self._insert_new(df, table)
			
		else:
			print(f'unknown mode: "{mode}"')
