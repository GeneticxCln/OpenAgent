# OpenAgent API Migration Guide

This guide helps you migrate between different versions of the OpenAgent API as new features are introduced and older versions are deprecated.

## API Versioning Overview

OpenAgent uses semantic versioning for its API endpoints with the following format: `MAJOR.MINOR` (e.g., `1.0`, `1.1`, `2.0`).

### Version Support Policy

- **Current Version**: The latest stable API version with full support
- **Supported Versions**: Older versions that continue to work but may receive limited updates
- **Deprecated Versions**: Versions scheduled for removal in future releases
- **Removed Versions**: Versions that no longer work and return errors

### How to Specify API Version

You can specify the API version in three ways:

1. **Accept Header** (Recommended):
   ```http
   Accept: application/json; version=1.0
   ```

2. **Custom Header**:
   ```http
   X-API-Version: 1.0
   ```

3. **Path Prefix** (Future):
   ```http
   GET /v1.0/chat
   ```

## Version History and Migration

### Version 1.0 (Current)

**Status**: ✅ Current stable version  
**Released**: January 2024  
**Support**: Full support with active development

#### Features
- Basic chat endpoints (`/chat`, `/chat/stream`)
- Code generation (`/code/generate`)
- Code analysis (`/code/analyze`)
- WebSocket streaming (`/ws`, `/ws/chat`)
- Authentication (`/auth/login`)
- Agent management (`/agents/*`)
- System information (`/system/info`)
- Health checks (`/health`, `/healthz`, `/readyz`)
- Metrics (`/metrics`)

#### Request/Response Format
```json
{
  "message": "Your question here",
  "agent": "default",
  "context": {}
}
```

### Version 1.1 (Planned)

**Status**: 🚧 In development  
**Expected**: Q2 2024  
**Migration**: Backward compatible

#### Planned Features
- Enhanced streaming with progress indicators
- Bulk operations for code analysis
- Advanced agent configuration
- Improved error responses with error codes
- File upload support for code analysis

#### Breaking Changes
None - fully backward compatible with 1.0

#### Migration Steps
No migration required. Version 1.0 clients will continue to work unchanged.

### Version 2.0 (Future)

**Status**: 📋 Planned  
**Expected**: Q4 2024  
**Migration**: Breaking changes

#### Planned Breaking Changes
- Unified response format across all endpoints
- Changed authentication flow
- Updated WebSocket message format
- Restructured agent management endpoints

#### Migration Timeline
- **6 months before release**: Version 1.x marked as deprecated
- **3 months before release**: Migration tools and guides released
- **Release day**: Version 1.x still supported but deprecated
- **6 months after release**: Version 1.x support ended

## Migration Checklist

### Before Upgrading

- [ ] Review the changelog for your target version
- [ ] Test your application with the new version in a staging environment
- [ ] Update any hardcoded version references
- [ ] Check for deprecated endpoint usage
- [ ] Update error handling for new error codes
- [ ] Verify WebSocket message format changes

### During Migration

- [ ] Update API version headers in your requests
- [ ] Implement new authentication flow (if changed)
- [ ] Update request/response models
- [ ] Test all endpoints with real data
- [ ] Update documentation and examples
- [ ] Train your team on new features

### After Migration

- [ ] Monitor application performance and error rates
- [ ] Remove old version handling code
- [ ] Update client SDKs and libraries
- [ ] Update API documentation

## Error Handling

### Version-Related Errors

**Unsupported Version**
```json
{
  "error": "unsupported_api_version",
  "message": "API version '0.9' is not supported",
  "supported_versions": ["1.0", "1.1"],
  "current_version": "1.0",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "abc123"
}
```

**Deprecated Version Warning**
```http
Warning: 299 - "API version 0.8 is deprecated and will be removed in a future release. Please upgrade to version 1.0."
```

## Testing Strategy

### Version Compatibility Testing

1. **Regression Testing**: Ensure old functionality still works
2. **Forward Compatibility**: Test with newer versions
3. **Error Handling**: Verify proper error responses
4. **Performance Testing**: Check for performance impacts

### Test Cases

```bash
# Test current version
curl -H "X-API-Version: 1.0" http://localhost:8000/chat

# Test deprecated version (should work with warning)
curl -H "X-API-Version: 0.9" http://localhost:8000/chat

# Test unsupported version (should fail)
curl -H "X-API-Version: 0.1" http://localhost:8000/chat
```

## SDK and Client Library Updates

### Python Client
```python
from openagent_client import OpenAgentClient

# Specify version
client = OpenAgentClient(
    base_url="http://localhost:8000",
    api_version="1.0"
)
```

### JavaScript Client
```javascript
const client = new OpenAgentClient({
  baseURL: 'http://localhost:8000',
  apiVersion: '1.0'
});
```

## Common Migration Issues

### Issue: Version Header Not Recognized
**Problem**: API version header is ignored  
**Solution**: Check header name and format. Use `X-API-Version: 1.0`

### Issue: Deprecated Endpoints
**Problem**: Endpoints return deprecation warnings  
**Solution**: Update to newer endpoints or handle warnings gracefully

### Issue: Changed Response Format
**Problem**: Client expects old response structure  
**Solution**: Update response parsing logic or use compatibility layer

## Getting Help

### Resources
- **API Documentation**: `/docs` endpoint
- **Version Info**: `GET /api/version`
- **Status Check**: `GET /api/status`
- **GitHub Issues**: Report migration problems
- **Community Forum**: Ask questions and share solutions

### Migration Support
- **Professional Services**: Available for complex migrations
- **Community Support**: Free help via GitHub Discussions
- **Documentation**: Comprehensive guides and examples
- **Tools**: Automated migration scripts (when available)

## Best Practices

### Version Management
1. **Pin Versions**: Always specify exact version numbers
2. **Test Early**: Start testing new versions as soon as they're available
3. **Gradual Migration**: Migrate non-critical systems first
4. **Monitor Warnings**: Pay attention to deprecation warnings
5. **Update Regularly**: Don't wait too long to upgrade

### Code Organization
```python
# Good: Version-specific handling
class OpenAgentClient:
    def __init__(self, api_version="1.0"):
        self.api_version = api_version
        self.headers = {"X-API-Version": api_version}
    
    def chat(self, message):
        if self.api_version >= "1.1":
            return self._chat_v1_1(message)
        else:
            return self._chat_v1_0(message)
```

### Error Handling
```python
try:
    response = client.chat("Hello")
except UnsupportedVersionError as e:
    logger.error(f"API version not supported: {e}")
    # Fallback to older version or show error to user
except DeprecatedVersionError as e:
    logger.warning(f"Using deprecated API version: {e}")
    # Continue but plan migration
```

## Changelog

### Version 1.0 (Current)
- Initial stable API release
- All core endpoints implemented
- WebSocket streaming support
- Full authentication system

### Version 0.9 (Deprecated)
- Beta API release
- Limited endpoint support
- **Migration to 1.0 required by March 2024**

For the most up-to-date migration information, always check the official changelog at: https://github.com/yourusername/OpenAgent/blob/main/CHANGELOG.md
