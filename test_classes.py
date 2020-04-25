class Animal:
	def __init__(self, animal_type):
		self.animal_type = animal_type
		self.num_legs = 0

	def get_type(self):
		print(self.animal_type)

	def set_legs(self, legs):
		self.num_legs = legs

class Track:
	def __init__(self, track_id):
		self.track_id = track_id
		self.track_positions = (0,0)

	def get_id(self):
		return self.track_id

	def set_position(self, position):
		self.track_positions = position



dog = Animal('dog')
cat = Animal('cat')
dog.get_type()
cat.get_type()
dog.set_legs(4)
