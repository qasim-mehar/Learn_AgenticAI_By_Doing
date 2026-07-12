from mcp.server.fastmcp import FastMCP

mcp=FastMCP("weather")

@mcp.tool()
def checkWeather():
    return "Weather here in islamabad is pretty cold"

if __name__=="__main__":
    mcp.run(transport="streamable-http")