-- Nothing but the tabular policy, to check that explore() really is it.
-- If this does not win the games mode 8 wins, the primitive is wrong and no
-- cleverness written on top of it will help.
local seen = level()
for i = 1, 400 do
  explore(100)
  if level() > seen then
    say("level " .. level() .. " at " .. steps() .. " actions")
    seen = level()
  end
  if steps() >= 40000 then break end
end
say("ended at level " .. level() .. " after " .. steps() .. " actions")
