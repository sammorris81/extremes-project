using(GLM)
using(DataFrames)

cd()  # set the working directory to homedir
cd("./repos-git/extremes-project/julia-tutorial")  # change dir relative to home
pwd()

# read in the data - note: readtable creates a dataframe as opposed to a matrix
data = readtable("gasoline.csv", header=true)


fit_lm = lm(octane_rating~component_1 + component_2 + component_3, data)
coef(fit_lm)

using RDatasets
iris = dataset("datasets", "iris")
iris[:id] = 1:size(iris, 1)

# create a dataframe that has multiple rows per item
d = stack(iris, [1:4])
# now get it back in the original form
widedf = unstack(d, :id, :variable, :value)

