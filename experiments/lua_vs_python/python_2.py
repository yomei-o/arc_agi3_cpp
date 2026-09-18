while True:
    objs = find_objects()
    targets = [o for o in objs if o.colour == 3]
    if not targets:
        break
    for t in targets:
        if level() != level():  # This check is flawed, need to track level
            break
        move_to(t.x, t.y)
    # Check level change after moving to all or breaking
    # Actually, the prompt says "stop as soon as level() changes"
    # So we need to check level before each action or in a loop
    # Let's restructure:
    # We need to move to each object of colour 3 in turn.
    # And stop as soon as level() changes.
    # Better approach:
    # Get all colour 3 objects.
    # Iterate through them.
    # Before moving to each, check if level changed? Or after?
    # "stop as soon as level() changes" implies checking frequently.
    # Let's assume we check level at the start of each iteration or after move.
    # But the list of objects might change if level changes.
    # Let's try:
    # current_level = level()
    # objs = find_objects()
    # targets = [o for o in objs if o.colour == 3]
    # for t in targets:
    #     if level() != current_level:
    #         break
    #     move_to(t.x, t.y)
    # However, the prompt says "Move to each object... in turn".
    # It doesn't say "only once". It might imply a continuous process?
    # "stop as soon as level() changes" suggests a loop that runs until level changes.
    # If I just do one pass, and level doesn
