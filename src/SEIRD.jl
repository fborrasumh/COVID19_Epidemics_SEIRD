using Pkg

Pkg.add("Distributions");
Pkg.add("LightGraphs");
Pkg.add("Plots");
Pkg.add("Printf");
Pkg.add("Random");
Pkg.add("Statistics");

using Distributions
using LightGraphs
using Plots
using Random
using Statistics

import Base: copy

# The possible states of an agent
@enum AgentState begin
    Susceptible
    Exposed
    Infected
    Recovered
    Dead
end

# The simulation state
mutable struct SimState
    # The graph
    graph::AbstractGraph
    # The individual states
    states::Array{AgentState}
    # Whether or not an individual dies when infected
    die::BitArray
    # The state counts
    counts::Dict{AgentState, Int}

    # Constructor
    SimState(
        g::AbstractGraph,
        s::Array{AgentState},
        cd::BitArray,
        c::Dict{AgentState, Int}) = new(g,s,cd,c)

    # Constructor
    function SimState(g_generator::Function, g_params, p_recover::Float64)
        # Create network
        graph = g_generator(g_params)
        # Everybody is susceptible initially
        states = fill(Susceptible, nv(graph))
        # Which agents can die
        die = rand(Bernoulli(1-p_recover), nv(graph))
        # Infect a random node
        states[rand(1:nv(graph))] = Infected
        # Initialize counts
        counts = Dict{AgentState, Int}()
        counts[Susceptible] = nv(graph)-1
        counts[Infected]    = 1
        counts[Exposed]     = 0
        counts[Recovered]   = 0
        counts[Dead]        = 0
        # Create fields
        new(graph, states, die, counts)
    end
end

# How to copy the simulation state
copy(s::SimState) = SimState(s.graph, copy(s.states), copy(s.die), copy(s.counts))

########################################

function seird_update(
    sim_state::SimState,    # The status of the simulation
    p_contagion_e::Float64, # S -> E : prob of contagion due to exposed
    p_contagion_i::Float64, # S -> E : prob of contagion due to infected
    p_incubation::Float64,  # E -> I : 1/p_incubation = average time of exposure
    p_symptomatic::Float64, # I <-> R, I <-> D : 1/p_symptomatic = average time of symptoms
    p_die::Float64          # I <-> D : fraction of death-risk individuals who effectively die
)
    # Copy old status into new
    #println("    OLD = ", sim_state.states)
    sim_state_new = copy(sim_state)
    #println("    COPIED = ", sim_state_new.states)

    # Update agents
    for agent in vertices(sim_state.graph)
        #println("    ", agent, ":", sim_state.states[agent])

        if sim_state.states[agent] == Susceptible
            # Susceptible state
            # We consider if either:
            # - It's not at risk of death, OR
            # - It's at death risk, and it's been exposed
            if (!sim_state_new.die[agent]) || (sim_state_new.die[agent] && rand() < p_die)
                for nbr in neighbors(sim_state.graph, agent)
                    #println("      ", nbr, ":", sim_state.states[nbr])
                    if sim_state.states[nbr] == Exposed || sim_state.states[nbr] == Infected
                        p_contagion = (sim_state.states[nbr] == Exposed) ? p_contagion_e : p_contagion_i
                        if rand() < p_contagion
                            sim_state_new.states[agent] = Exposed
                            sim_state_new.counts[Susceptible] -= 1
                            sim_state_new.counts[Exposed] += 1
                            #println("        CHANGED to Exposed")
                            break
                        end
                    end
                end
            end

        elseif sim_state.states[agent] == Exposed
            if rand() < p_incubation
                sim_state_new.states[agent] = Infected
                sim_state_new.counts[Exposed] -= 1
                sim_state_new.counts[Infected] += 1
                #println("        CHANGED to Infected")
            end

        elseif sim_state.states[agent] == Infected
            if rand() < p_symptomatic
                sim_state_new.counts[Infected] -= 1
                if !sim_state_new.die[agent]
                    sim_state_new.states[agent] = Recovered
                    sim_state_new.counts[Recovered] += 1
                    #println("        CHANGED to Recovered")
                else
                    sim_state_new.states[agent] = Dead
                    sim_state_new.counts[Dead] += 1
                    #println("        CHANGED to Dead")
                end
            end
        end
    end
    #println("    NEW = ", sim_state_new.states)
    #println("    ", sim_state_new.counts)

    return sim_state_new
end

########################################

function seird_simulate(
    g_generator::Function,   # network generation function
    g_params,                # parameters for network generation function
    p_contagion_e::Float64,  # S -> E : prob of contagion due to exposed
    p_contagion_i::Float64,  # S -> E : prob of contagion due to infected
    t_incubation::Int,       # E -> I : average time of exposure
    t_symptomatic::Int,      # I <-> R, I <-> D : average time of symptoms
    p_recover::Float64,      # I <-> R, I <-> D : fraction of population that can recover if infected (others die)
    f_die::Function,         # I <-> D : fraction of death-risk individuals who effectively die
    f_die_params,            # parameters for f_die()
    t_max::Int,              # maximum time steps
    n_reps::Int,             # repetitions
    seed::Int                # random seed
) :: Dict{AgentState, Array{Float64}}
    # Set random seed
    Random.seed!(seed)

    # Dictionary to store simulation data
    # agent state
    #   repetition
    #     time step
    data = Dict{AgentState, Array{Float64}}()
    for s in instances(AgentState)
        data[s] = Array{Float64}(undef,t_max,n_reps)
    end

    # Go through the repetitions
    for rep in 1:n_reps
        #println("\n\nnew rep")

        # Create simulation state
        sim_state = SimState(g_generator, g_params, p_recover)
        # Save the initial state counts
        for s in instances(AgentState)
            data[s][1,rep] = sim_state.counts[s] / nv(sim_state.graph)
        end

        # Time updates
        p_incubation = 1.0 / t_incubation
        p_symptomatic = 1.0 / t_symptomatic
        for t in 2:t_max
            #println("  new t")
            # Update the network
            sim_state = seird_update(
                sim_state,
                p_contagion_e,
                p_contagion_i,
                p_incubation,
                p_symptomatic,
                f_die(t, f_die_params))
            #println("    RETURNED = ", sim_state.states)

            # Save the state counts
            for s in instances(AgentState)
                data[s][t,rep] = sim_state.counts[s] / nv(sim_state.graph)
            end
        end
    end

    return data
end

########################################

function seird_plot(data::Dict{AgentState, Array{Float64}}, n_candie::Float64, ttl::String)
    colors = Dict(
        Susceptible => :gray,
        Exposed     => :lightskyblue,
        Infected    => :navy,
        Recovered   => :forestgreen,
        Dead        => :tomato
    )
    t_max = size(data[Susceptible], 1)
    t_span = range(1, stop=t_max)
    atrisk = Plots.text("at risk of dying", :right, :bottom, 10)
    plt = plot(t_span,
               fill(n_candie,length(t_span)),
               show=true,
               label=nothing,
               legend=:right,
               linestyle=:dash,
               linewidth=1,
               seriescolor=:black,
               ann=(t_max, n_candie, atrisk),
               title=ttl,
               titlefontsize=10)
    for s in instances(AgentState)
        m = mean(data[s], dims=2)
        st = Statistics.std(data[s]; mean=m, dims=2)
        plot!(plt, t_span, m, ribbon=st, label=s, linewidth=2, seriescolor=colors[s])
    end
    return plt
end

########################################
