local objects = find_objects()
for _, obj in ipairs(objects) do
    if level() ~= level() then break end -- check if level changed before moving
    if obj.colour == 3 then
        move_to(obj.x, obj.y)
    end
    if level() ~= level() then break end -- check if level changed after moving
end
