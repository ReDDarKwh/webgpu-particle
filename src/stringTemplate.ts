export default function stringTemplate(str: string, obj: any) {
    return str.replace(/\${(\w+)}/gm, (_match, p1) => {
        return obj[p1];
    });
}