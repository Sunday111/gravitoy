#include "gravitoy/app.hpp"
#include "klgl/error_handling.hpp"

namespace klgl::gravitoy
{
void Main()
{
    GravitoyApp app;
    app.Run();
}
}  // namespace klgl::gravitoy

int main()
{
    klgl::ErrorHandling::InvokeAndCatchAll(klgl::gravitoy::Main);
    return 0;
}
