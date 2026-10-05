objects = find_objects()
objects = [obj for obj in objects if obj.area <= 4]
objects.sort(key=lambda obj: obj.area)
for obj in objects:
    click(obj.x, obj.y)
