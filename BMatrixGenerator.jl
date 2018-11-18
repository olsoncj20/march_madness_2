
# m = Model(solver = IpoptSolver(print_level=0))

data = readtable("FILL IN CSV NAME", header=true)
# x_1 = data[:,1]
# x_2 = data[:,2]
# y_1 = data[:,3]
# y_2 = data[:,4]

@variables m begin
    B # B matrix   
    c1 # c1 = sum of column 1 = number of wins in round 1
    c2
    c3
    c4
    c5
    c6
end

# Constraint for number of wins per round
@constraint(m, c1 == 32)
@constraint(m, c2 == 16)
@constraint(m, c3 == 8)
@constraint(m, c4 == 4)
@constraint(m, c5 == 2)
@constraint(m, c6 == 1)

# Constraint for once one row gets a zero, zero for the rest of the row
for t in 1:64
    for r in 1:5
        @constraint(m, B[t,r] >= B[t,r+1])
    end
end

# Pairwise Constraints

# Round 1
for t in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,
    43,45,47,49,51,53,55,57,59,61,63]
    @constraint(m, B[t,1] + B[t+1,1] == 1)
end

# Round 2
for t in [1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61]
    @constraint(m, B[t,2] + B[t+1,2] + B[t+2,2] + B[t+3,2] == 1)
end

# Round 3
for t in [1,9,17,25,33,41,49,57]
    @constraint(m, B[t,2] + B[t+1,2] + B[t+2,2] + B[t+3,2] + 
        B[t+4,2] + B[t+5,2] + B[t+6,2] + B[t+7,2]== 1)
end

# Round 4
@constraint(m, sum(B[t,4] for t in 1:16) == 1)  
@constraint(m, sum(B[t,4] for t in 17:32) == 1) 
@constraint(m, sum(B[t,4] for t in 33:48) == 1)  
@constraint(m, sum(B[t,4] for t in 49:64) == 1)

# Round 5
@constraint(m, sum(B[t,5] for t in 1:32) == 1)  
@constraint(m, sum(B[t,5] for t in 33:64) == 1) 

# Round 6
@constraint(m, sum(B[t,6] for t in 1:64) == 1)  

# @NLobjective(m, Min, w1^2 + w2^2)
# solve(m)
# println("The objective value is: ", getobjectivevalue(m))
# println("The value of w1 is: ", getvalue(w1))
# println("The value of w2 is: ", getvalue(w2))
# println("The value of b is: ", getvalue(b))
