using DataFrames
using JuMP
using Gurobi
using CSV

function generate_brackets(Q, num_overlap, brackets, num_brackets, S)
	m = Model(solver=GurobiSolver(OutputFlag=0))

	# Variable for generate_brackets
	@variable(m, B[i=1:64, j=1:6], Bin)

	# Non-increasing rows
	for t in 1:64
		for r in 1:5
			@constraint(m, B[t,r]>=B[t,r+1])
		end
	end

	# Feasability constraint
	@constraint(m, feas[r=1:6,block=0:2^(6-r)-1], sum{B[j,r], j=2^r*block+1:2^r*block+2^r}==1)

	# Overlap constraint
	for i=1:num_brackets
		bracket = brackets[:,6*i-5:6*i]
		@constraint(m, sum{sum{bracket[k,j]*B[k,j], k=1:64}, j=1:6} <= num_overlap)
	end

	# Objective
	@objective(m, Max, sum{S[j]*sum{B[k,j]*Q[k,j], k=1:64}, j=1:6})

	# Solve it
	print("Solving"*string(num_brackets)*"...")
	status = solve(m)

	if status==:Optimal
		println("Complete")
		bracket = []
		for t in 1:64
			if getvalue(B[t,1]) >= 0.9 && getvalue(B[t,1]) <= 1.1
				bracket = vcat(bracket, fill(1,1))
			else
				bracket = vcat(bracket, fill(0,1))
			end
		end	
		for r in 2:6
			round_data = []
			for t in 1:64
				if getvalue(B[t,r]) >= 0.9 && getvalue(B[t,r]) <= 1.1
					round_data = vcat(round_data, fill(1,1))
				else
					round_data = vcat(round_data, fill(0,1))
				end
			end
			bracket = hcat(bracket, round_data)
		end
		return(bracket)
	end
	return([])
end


function create_brackets(num_brackets, num_overlap, path_q_matrix, formulation, output_path)
	q = CSV.read(path_q_matrix, header=true)

	q_copy = []
	for j=1:64
		q_copy = vcat(q_copy, q[j, 2])
	end

	for i=3:7
		q_col = []
		for j=1:64
			q_col = vcat(q_col, q[j, i])
		end
		q_copy = hcat(q_copy, q_col)
	end

	empty_bracket = hcat(zeros(Int, 64), zeros(Int, 64), zeros(Int, 64), zeros(Int, 64), zeros(Int, 64), zeros(Int, 64))

	num_brackets_so_far = 0

	S = [10,20,40,80,160,320]

	bracket = formulation(q_copy, num_overlap, empty_bracket, num_brackets_so_far, S)
	if bracket == []
		return
	end
	num_brackets_so_far += 1

	brackets = bracket

	for i=1:num_brackets-1
		try
			the_bracket = formulation(q_copy, num_overlap, brackets, num_brackets_so_far, S)
			num_brackets_so_far += 1
			brackets = hcat(brackets, the_bracket)
		catch
			break
		end
	end

	# output to CSV
	for i=1:num_brackets
		bracket = brackets[:,6*i-5:6*i]
		df = DataFrame(bracket)
		CSV.write(output_path*"_"*string(i)*".csv", df)
	end
end


num_brackets = 50
opt = generate_brackets

for year in ["2013"]
	for overlap=30:60
		path = "q_matrix_noise"*year*".csv"
		println(year)
		println(overlap)
		download_path = "brackets/"*year*"_noise_"*string(overlap)
		create_brackets(num_brackets, overlap, path, opt, download_path)
	end
end








