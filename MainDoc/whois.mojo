import python

python.def lookup_ipv6_cidr(ip: str):
    from ipwhois import IPWhois
    import ipaddress

    ip = ipaddress.ip_address(ip)
    obj = IPWhois(str(ip))
    res = obj.lookup_rdap()
    return res.get("network", {}).get("cidr", "CIDR not found")

cidr = lookup_ipv6_cidr("2a06:98c1:54::1a:df76")
print("CIDR block:", cidr)