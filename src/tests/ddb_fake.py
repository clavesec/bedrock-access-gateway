"""A minimal DynamoDB fake for the taint/budget stores.

Implements exactly the request shapes ``api.taint`` and ``api.budget`` use —
``get_item`` and the two conditional ``update_item`` expressions — with real
condition evaluation, so the state-machine tests exercise the actual
mutual-exclusion semantics instead of mirroring them. Any expression this
fake does not recognize fails the test loudly: drift between the modules and
the fake must surface, not silently pass.
"""


class ConditionalCheckFailedException(Exception):
    """Named to match botocore's exception class name — the modules detect
    conditional failures via ``type(exc).__name__``."""


class FakeDynamoDB:
    def __init__(self, fail_with: Exception | None = None):
        # table name -> pk -> item (attribute name -> typed value dict)
        self.tables: dict[str, dict[str, dict]] = {}
        self.fail_with = fail_with
        self.calls: list[tuple[str, dict]] = []

    def _table(self, name: str) -> dict:
        return self.tables.setdefault(name, {})

    def get_item(self, *, TableName, Key, ConsistentRead=False):
        self.calls.append(("get_item", {"TableName": TableName, "Key": Key}))
        if self.fail_with is not None:
            raise self.fail_with
        item = self._table(TableName).get(Key["pk"]["S"])
        return {"Item": dict(item)} if item is not None else {}

    def update_item(
        self,
        *,
        TableName,
        Key,
        UpdateExpression,
        ConditionExpression,
        ExpressionAttributeNames,
        ExpressionAttributeValues,
        ReturnValuesOnConditionCheckFailure=None,
    ):
        self.calls.append(("update_item", {"TableName": TableName, "Key": Key}))
        if self.fail_with is not None:
            raise self.fail_with
        assert ExpressionAttributeNames == {"#ttl": "ttl"}, ExpressionAttributeNames
        pk = Key["pk"]["S"]
        table = self._table(TableName)
        item = table.get(pk, {})
        vals = ExpressionAttributeValues

        def _conditional_failure(message):
            exc = ConditionalCheckFailedException(message)
            # Mirror botocore: with ReturnValuesOnConditionCheckFailure=
            # ALL_OLD the refused write carries the pre-image on
            # exc.response["Item"].
            exc.response = (
                {"Item": dict(item)} if ReturnValuesOnConditionCheckFailure == "ALL_OLD" else {}
            )
            return exc

        if ConditionExpression == "attribute_not_exists(tainted_class) OR tainted_class = :cls":
            if "tainted_class" in item and item["tainted_class"]["S"] != vals[":cls"]["S"]:
                raise _conditional_failure("taint class conflict")
        elif ConditionExpression == "attribute_not_exists(n) OR n < :limit":
            if "n" in item and int(item["n"]["N"]) >= int(vals[":limit"]["N"]):
                raise _conditional_failure("budget limit reached")
        else:
            raise AssertionError(f"unrecognized ConditionExpression: {ConditionExpression!r}")

        if UpdateExpression == "SET tainted_class = :cls, #ttl = :ttl":
            item = dict(item)
            item["tainted_class"] = {"S": vals[":cls"]["S"]}
            item["ttl"] = {"N": vals[":ttl"]["N"]}
        elif UpdateExpression == "ADD n :one SET #ttl = :ttl":
            item = dict(item)
            n = int(item.get("n", {"N": "0"})["N"]) + int(vals[":one"]["N"])
            item["n"] = {"N": str(n)}
            item["ttl"] = {"N": vals[":ttl"]["N"]}
        else:
            raise AssertionError(f"unrecognized UpdateExpression: {UpdateExpression!r}")

        table[pk] = item
        return {}
