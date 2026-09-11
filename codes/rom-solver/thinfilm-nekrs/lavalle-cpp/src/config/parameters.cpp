#include "config/parameters.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>


namespace
{

// -------------------------------------------------------------
// Remove whitespace from beginning/end of a string
// -------------------------------------------------------------

std::string trim(const std::string& s)
{
    const auto first = s.find_first_not_of(" \t\r\n");

    if (first == std::string::npos)
        return "";

    const auto last = s.find_last_not_of(" \t\r\n");

    return s.substr(first, last - first + 1);
}


// -------------------------------------------------------------
// Read input file into a key/value map
// -------------------------------------------------------------

std::unordered_map<std::string, std::string>
read_key_value_file(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error(
            "Could not open input file: " + filename
        );

    std::unordered_map<std::string, std::string> values;

    std::string line;
    int line_number = 0;

    while (std::getline(file, line))
    {
        ++line_number;

        // Remove comments
        const auto comment_pos = line.find('#');

        if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos);

        line = trim(line);

        // Skip blank lines
        if (line.empty())
            continue;

        // Find '='
        const auto equals_pos = line.find('=');

        if (equals_pos == std::string::npos)
        {
            throw std::runtime_error(
                "Invalid input at line "
                + std::to_string(line_number)
                + ": expected 'key = value'"
            );
        }

        const std::string key =
            trim(line.substr(0, equals_pos));

        const std::string value =
            trim(line.substr(equals_pos + 1));

        if (key.empty() || value.empty())
        {
            throw std::runtime_error(
                "Invalid input at line "
                + std::to_string(line_number)
            );
        }

        // Do not silently overwrite duplicate parameters
        if (values.find(key) != values.end())
        {
            throw std::runtime_error(
                "Duplicate input parameter: " + key
            );
        }

        values[key] = value;
    }

    return values;
}


// -------------------------------------------------------------
// Required scalar
// -------------------------------------------------------------

double get_double(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key
)
{
    const auto it = values.find(key);

    if (it == values.end())
    {
        throw std::runtime_error(
            "Missing required input parameter: " + key
        );
    }

    try
    {
        return std::stod(it->second);
    }
    catch (...)
    {
        throw std::runtime_error(
            "Could not convert parameter '" + key
            + "' to double: " + it->second
        );
    }
}


// -------------------------------------------------------------
// Required integer
// -------------------------------------------------------------

int get_int(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key
)
{
    const auto it = values.find(key);

    if (it == values.end())
    {
        throw std::runtime_error(
            "Missing required input parameter: " + key
        );
    }

    try
    {
        return std::stoi(it->second);
    }
    catch (...)
    {
        throw std::runtime_error(
            "Could not convert parameter '" + key
            + "' to integer: " + it->second
        );
    }
}


// -------------------------------------------------------------
// Check whether parameter exists
// -------------------------------------------------------------

bool has_key(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key
)
{
    return values.find(key) != values.end();
}


// -------------------------------------------------------------
// Make sure exactly one of dimensional/nondimensional
// quantities was supplied.
// -------------------------------------------------------------

void require_exactly_one(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& nondim_key,
    const std::string& dim_key
)
{
    const bool has_nondim = has_key(values, nondim_key);
    const bool has_dim = has_key(values, dim_key);

    if (has_nondim && has_dim)
    {
        throw std::runtime_error(
            "Specify only one of '" + nondim_key
            + "' or '" + dim_key + "', not both."
        );
    }

    if (!has_nondim && !has_dim)
    {
        throw std::runtime_error(
            "Missing parameter: specify either '"
            + nondim_key + "' or '" + dim_key + "'."
        );
    }
}

} // namespace


Parameters Parameters::from_file(const std::string& filename)
{
    Parameters p;

    const auto values = read_key_value_file(filename);

    // =========================================================
    // Read quantities that do not have dimensional aliases
    // =========================================================

    p.in.fcut_dim =
        get_double(values, "fcut_dim");

    p.in.MNOISE =
        get_int(values, "MNOISE");

    p.in.F0_noise =
        get_double(values, "F0_noise");

    p.in.N =
        get_int(values, "N");

    p.in.Re_input =
        get_double(values, "Re_input");

    p.in.theta_deg =
        get_double(values, "theta_deg");

    p.in.rho =
        get_double(values, "rho");

    p.in.mu =
        get_double(values, "mu");

    p.in.Ca_input =
        get_double(values, "Ca_input");


    // =========================================================
    // Validate dimensional/nondimensional alternatives
    // =========================================================

    require_exactly_one(values, "L", "Ldim");
    require_exactly_one(values, "t_end", "t_end_dim");
    require_exactly_one(values, "dt", "dt_dim");
    require_exactly_one(values, "t_save", "t_save_dim");


    // =========================================================
    // Physical scales
    //
    // These MUST be calculated before converting L/time because
    // hN and TN define those conversions.
    // =========================================================

    constexpr double pi =
        3.14159265358979323846;

    p.theta =
        p.in.theta_deg * pi / 180.0;

    p.nu =
        p.in.mu / p.in.rho;

    p.We =
        p.in.Ca_input * p.in.Re_input;

    p.hN =
        std::cbrt(
            3.0
            * p.in.Re_input
            * p.nu
            * p.nu
            /
            (p.g * std::sin(p.theta))
        );

    p.uN =
        p.nu
        * p.in.Re_input
        / p.hN;

    p.TN =
        p.hN / p.uN;

    p.Re =
        p.uN
        * p.hN
        / p.nu;

    p.F2 =
        p.uN
        * p.uN
        /
        (p.g * p.hN);

    p.Ca =
        p.We / p.Re;

    p.gamma =
        p.in.rho
        * p.hN
        * p.uN
        * p.uN
        / p.We;

    p.fcut_noise =
        p.in.fcut_dim * p.TN;


    // =========================================================
    // DOMAIN LENGTH
    // =========================================================

    if (has_key(values, "L"))
    {
        // Input already nondimensional
        p.L =
            get_double(values, "L");

        // Recover dimensional value for output/metadata
        p.Ldim =
            p.L * p.hN;
    }
    else
    {
        // Input is dimensional [m]
        p.Ldim =
            get_double(values, "Ldim");

        // Convert to nondimensional
        p.L =
            p.Ldim / p.hN;
    }


    // =========================================================
    // END TIME
    // =========================================================

    if (has_key(values, "t_end"))
    {
        p.t_end =
            get_double(values, "t_end");

        p.t_end_dim =
            p.t_end * p.TN;
    }
    else
    {
        p.t_end_dim =
            get_double(values, "t_end_dim");

        p.t_end =
            p.t_end_dim / p.TN;
    }


    // =========================================================
    // TIMESTEP
    // =========================================================

    if (has_key(values, "dt"))
    {
        p.dt =
            get_double(values, "dt");

        p.dt_dim =
            p.dt * p.TN;
    }
    else
    {
        p.dt_dim =
            get_double(values, "dt_dim");

        p.dt =
            p.dt_dim / p.TN;
    }


    // =========================================================
    // SAVE INTERVAL
    // =========================================================

    if (has_key(values, "t_save"))
    {
        p.t_save =
            get_double(values, "t_save");

        p.t_save_dim =
            p.t_save * p.TN;
    }
    else
    {
        p.t_save_dim =
            get_double(values, "t_save_dim");

        p.t_save =
            p.t_save_dim / p.TN;
    }


    // =========================================================
    // Grid
    // =========================================================

    p.ng = 1;

    p.Nx =
        p.in.N + 2 * p.ng;

    p.dx =
        p.L / static_cast<double>(p.in.N);


    // =========================================================
    // Adjust dt so final timestep lands exactly on t_end
    // =========================================================

    p.nsteps =
        static_cast<int>(
            std::ceil(p.t_end / p.dt)
        );

    p.dt =
        p.t_end
        / static_cast<double>(p.nsteps);

    // dt may have changed slightly, so recompute dimensional dt
    p.dt_dim =
        p.dt * p.TN;


    // =========================================================
    // Output cadence
    // =========================================================

    p.save_every =
        std::max(
            1,
            static_cast<int>(
                std::round(
                    p.t_save / p.dt
                )
            )
        );

    p.progress_every =
        std::max(
            1,
            p.nsteps / 100
        );


    // =========================================================
    // Basic validation
    // =========================================================

    if (p.L <= 0.0)
        throw std::runtime_error("L must be positive.");

    if (p.t_end <= 0.0)
        throw std::runtime_error("t_end must be positive.");

    if (p.dt <= 0.0)
        throw std::runtime_error("dt must be positive.");

    if (p.in.N <= 0)
        throw std::runtime_error("N must be positive.");

    if (p.hN <= 0.0 || p.TN <= 0.0)
        throw std::runtime_error(
            "Invalid dimensional scales hN or TN."
        );


    return p;
}


