local objects = find_objects()
table.sort(objects, function(a, b) return a.area < b.area end)
for _, obj in ipairs(objects) do
    if obj.area <= 4 then
        click(obj.x, obj.y)
    end
end
